"""Configurable deduplication for structured memories.

The deduplicator runs **after** extraction (or as a standalone sweep) and
compares a memory's rows to find and compact duplicates. It owns no state on
the memories themselves — memories stay pure "add" operations and return what
they added from :meth:`~character_memory.memory.base.Memory.apply_extraction`;
the agent hands those items to the deduplicator as a post-step.

A pair is considered a duplicate when it passes every *enabled* gate, evaluated
in escalating order: exact match → similarity → LLM judge. If ``consolidate``
is on, a confirmed duplicate is rewritten into a single merged entry instead of
the newer one being dropped.

Contradiction resolution is an additional gate driven by each memory's
:class:`~character_memory.config.ContradictionPolicy`. When dedup finds no
duplicate for a candidate, the same ranked candidates are re-checked against a
*lower* similarity bar: if two entries clash (assert incompatible facts about
the same point in time), the older row's text is overwritten with the newer
row's and the newer row is dropped. Memories decide their own policy via
:meth:`StructuredMemory.contradiction_policy`; the default is disabled.

The dedup behaviours are toggled by :class:`~character_memory.config.DedupConfig`
flags; no LLM is required unless ``llm_judge``, ``consolidate`` or a memory's
contradiction policy are on (those stages degrade to no-ops when no
``LLMClient`` was supplied).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..config import ContradictionPolicy, DedupConfig
from ..llm.base import LLMClient
from ..llm.embedding_base import EmbeddingProvider
from ..prompts import PromptConfig
from .base import MemoryItem
from .structured import StructuredMemory

# JSON schemas for the structured-LLM calls (the prompts themselves live on
# `PromptConfig` so they are overridable like every other prompt in the system).
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"same": {"type": "boolean"}},
    "required": ["same"],
}

_CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}

_CONTRADICT_SCHEMA = {
    "type": "object",
    "properties": {"contradicts": {"type": "boolean"}},
    "required": ["contradicts"],
}


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _fmt_ts(ts: Any) -> str:
    """Render a `created_at` epoch float as a short readable stamp."""
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts)))
    except (TypeError, ValueError, OverflowError):
        return "unknown time"


@dataclass
class DedupReport:
    """Summary of a dedup pass (per memory)."""

    checked: int = 0          # items/rows considered
    skipped: int = 0          # duplicates dropped (no consolidation)
    merged: int = 0           # pairs consolidated into one
    resolved: int = 0         # contradictions: older text overwritten by newer
    removed_ids: list[int] = field(default_factory=list)
    updated_ids: list[int] = field(default_factory=list)


class Deduplicator:
    """Exact / similarity / LLM-judge deduplication with optional consolidation.

    Two entry points:

    * `dedup_items` — compact a batch of freshly-added items against the
      memory's existing rows (the post-extraction step).
    * `sweep` — compact a whole memory (optionally one user) in place.

    The LLM prompts are taken from ``prompts`` (a
    :class:`~character_memory.prompts.PromptConfig`); pass a customized one to
    override the judge / consolidate wording.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        llm: Optional[LLMClient] = None,
        config: Optional[DedupConfig] = None,
        prompts: Optional[PromptConfig] = None,
    ) -> None:
        self.embedder = embedder
        self.llm = llm
        self.config = config or DedupConfig()
        prompts = prompts or PromptConfig()
        self.judge_prompt = prompts.dedup_judge
        self.consolidate_prompt = prompts.dedup_consolidate
        self.contradict_prompt = prompts.dedup_contradict
        self._warned_no_llm = False

    # ------------------------------------------------------------------ utils
    def _llm_available(self, stage: str) -> bool:
        """True if the LLM is wired; otherwise warn once and return False."""
        if self.llm is not None:
            return True
        if not self._warned_no_llm:
            warnings.warn(
                f"DedupConfig.{stage} is enabled but no LLMClient was provided "
                f"to Deduplicator; the {stage} stage will be skipped.",
                stacklevel=2,
            )
            self._warned_no_llm = True
        return False

    def _embed_normalized(self, texts: list[str]) -> np.ndarray:
        """Embed and L2-normalize, returning a ``(n, dim)`` float32 matrix."""
        if not texts:
            return np.zeros((0, self.embedder.dim), dtype="float32")
        vecs = self.embedder.embed(list(texts)).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vecs / norms

    def _retrieve_candidates(
        self,
        memory: StructuredMemory,
        query: str,
        user_id: str,
        rows: list[dict[str, Any]],
        exclude_ids: Optional[set[int]] = None,
        pool: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Narrow candidate rows via the memory's RAG index (no full re-embed).

        ``exclude_ids`` (if given) are removed from the hits — used by
        :meth:`dedup_items` to avoid an item matching itself in the index.
        ``pool`` overrides ``DedupConfig.candidate_pool``; contradiction
        resolution asks for a wider net than plain dedup.
        """
        rows_by_id = {int(r["id"]): r for r in rows}
        exclude = exclude_ids or set()
        base_pool = pool if pool is not None else self.config.candidate_pool
        # Over-fetch so exclusion of a few top hits still yields candidates.
        k = min(base_pool + len(exclude), len(rows_by_id))
        if k <= 0:
            return []
        try:
            hits = memory.hybrid.search(query, k=k, where={"user_id": user_id})
        except Exception:
            return []
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for h in hits:
            rid = h.metadata.get("id")
            if rid is None:
                continue
            rid = int(rid)
            if rid in exclude or rid in seen or rid not in rows_by_id:
                continue
            seen.add(rid)
            out.append(rows_by_id[rid])
        return out

    def _judge(self, text_a: str, text_b: str) -> bool:
        """LLM gate: True if the two entries convey the same information."""
        messages = [
            {"role": "system", "content": self.judge_prompt},
            {"role": "user", "content": f"Entry A:\n{text_a}\n\nEntry B:\n{text_b}"},
        ]
        try:
            result = self.llm.chat_structured(messages, _JUDGE_SCHEMA)  # type: ignore[union-attr]
            return bool(result.get("same", False))
        except Exception:
            return False

    def _consolidate(self, text_a: str, text_b: str) -> Optional[str]:
        """LLM merge of two near-duplicates into one entry, or None on failure."""
        messages = [
            {"role": "system", "content": self.consolidate_prompt},
            {"role": "user", "content": f"Entry A:\n{text_a}\n\nEntry B:\n{text_b}"},
        ]
        try:
            result = self.llm.chat_structured(messages, _CONSOLIDATE_SCHEMA)  # type: ignore[union-attr]
            merged = (result.get("text") or "").strip()
            return merged or None
        except Exception:
            return None

    def _contradicts(
        self,
        text_a: str,
        text_b: str,
        ts_a: Any = None,
        ts_b: Any = None,
        show_ts: bool = True,
    ) -> bool:
        """LLM gate: True if the two entries cannot both be true at once.

        Timestamps are included when ``show_ts`` is set so the judge can tell a
        genuine clash from a change over time.
        """
        if show_ts:
            body = (
                f"Entry A (recorded {_fmt_ts(ts_a)}):\n{text_a}\n\n"
                f"Entry B (recorded {_fmt_ts(ts_b)}):\n{text_b}"
            )
        else:
            body = f"Entry A:\n{text_a}\n\nEntry B:\n{text_b}"
        messages = [
            {"role": "system", "content": self.contradict_prompt},
            {"role": "user", "content": body},
        ]
        try:
            result = self.llm.chat_structured(messages, _CONTRADICT_SCHEMA)  # type: ignore[union-attr]
            return bool(result.get("contradicts", False))
        except Exception:
            return False

    def _merge_rows(
        self,
        survivor: dict[str, Any],
        loser: dict[str, Any],
        text_col: str,
        merged_text: Optional[str] = None,
    ) -> dict[str, Any]:
        """Combine two rows' metadata into the survivor (sensible defaults).

        ``importance``/``confidence`` take the max, ``recall_count`` sums,
        ``created_at`` keeps the earliest, ``last_recalled`` the most recent.
        Other columns keep the survivor's value. ``merged_text`` (if given)
        overwrites the text column.
        """
        merged = dict(survivor)
        merged["importance"] = max(
            float(survivor.get("importance") or 0.0),
            float(loser.get("importance") or 0.0),
        )
        merged["recall_count"] = int(survivor.get("recall_count") or 0) + int(
            loser.get("recall_count") or 0
        )
        created = [x for x in (survivor.get("created_at"), loser.get("created_at")) if x is not None]
        if created:
            merged["created_at"] = min(created)
        recalled = [x for x in (survivor.get("last_recalled"), loser.get("last_recalled")) if x is not None]
        merged["last_recalled"] = max(recalled) if recalled else None
        if "confidence" in survivor:
            merged["confidence"] = max(
                float(survivor.get("confidence") or 0.0),
                float(loser.get("confidence") or 0.0),
            )
        if merged_text is not None:
            merged[text_col] = merged_text
        return merged

    # ----------------------------------------------------- duplicate test

    def _find_duplicate(
        self,
        memory: StructuredMemory,
        text: str,
        user_id: str,
        existing_rows: list[dict[str, Any]],
        exclude_ids: Optional[set[int]] = None,
        policy: Optional[ContradictionPolicy] = None,
        current_ts: Any = None,
    ) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
        """Run the escalating gates against ``existing_rows``.

        Returns ``(row, merged_text, reason)``:

        * ``row`` — the existing row this ``text`` should fold into (or None).
        * ``merged_text`` — the text to write back onto ``row`` (or None to
          leave ``row``'s text untouched). For a duplicate consolidation it is
          the LLM-merged sentence; for a contradiction it is the newer text
          (``text``) verbatim.
        * ``reason`` — ``"dup"`` (duplicate, ``row`` survives, ``text``'s row
          is dropped), ``"contradict"`` (same survivor/loser direction, but
          ``row``'s text is overwritten with the newer ``text``), or None.

        ``exclude_ids`` is forwarded to candidate retrieval so an item never
        matches itself. ``policy`` enables the contradiction gate and is taken
        from :meth:`StructuredMemory.contradiction_policy` by the caller.
        ``current_ts`` is the ``created_at`` of the row ``text`` came from, so
        the contradiction judge can weigh the timestamps.
        """
        cfg = self.config
        text_col = memory.text_column
        exclude = exclude_ids or set()
        contradict_on = bool(policy and policy.enabled)

        # Gate 1: exact match (cheapest).
        if cfg.exact:
            norm = _normalize(text)
            for r in existing_rows:
                if int(r["id"]) in exclude:
                    continue
                if _normalize(str(r.get(text_col, ""))) == norm:
                    return r, None, "dup"

        # Similarity retrieval shared by dedup + contradiction. Contradictions
        # sit at lower cosine, so widen the pool when that gate is on.
        sim_on = cfg.similarity_threshold is not None
        contra_on = contradict_on and policy.similarity_threshold is not None  # type: ignore[union-attr]
        if (sim_on or contra_on) and existing_rows:
            pool = (
                max(self.config.candidate_pool, policy.candidate_pool)  # type: ignore[union-attr]
                if contra_on
                else None
            )
            candidates = self._retrieve_candidates(
                memory, text, user_id, existing_rows, exclude_ids=exclude_ids, pool=pool
            )
            if candidates:
                cand_texts = [str(r.get(text_col, "")) for r in candidates]
                vecs = self._embed_normalized([text] + cand_texts)
                cand_vec = vecs[0]
                sims = vecs[1:] @ cand_vec
                best_j = int(np.argmax(sims))
                best_sim = float(sims[best_j])
                dup_row = candidates[best_j]
                dup_text = cand_texts[best_j]
                dup_ts = dup_row.get("created_at")

                # --- Dedup gate (high bar) ---
                is_dup = False
                if sim_on and best_sim >= cfg.similarity_threshold:  # type: ignore[operator]
                    # Gate 3: LLM judge (optional confirmation). When off, the
                    # candidate is a tentative duplicate — disambiguated below.
                    if cfg.llm_judge and self._llm_available("llm_judge"):
                        is_dup = self._judge(text, dup_text)
                    else:
                        is_dup = True

                # A tentative duplicate may actually be a contradiction
                # ("doctor" vs "engineer" are close but clash). When the policy
                # is on, ask the contradiction gate before calling it a dup.
                if is_dup and contra_on and self._llm_available("contradict"):
                    if self._contradicts(
                        dup_text, text, dup_ts, current_ts,
                        policy.show_timestamps,  # type: ignore[union-attr]
                    ):
                        return dup_row, text, "contradict"

                if is_dup:
                    # Confirmed duplicate — consolidate or just flag it.
                    if cfg.consolidate and self._llm_available("consolidate"):
                        merged_text = self._consolidate(dup_text, text)
                        if merged_text:
                            return dup_row, merged_text, "dup"
                    return dup_row, None, "dup"

                # --- Contradiction gate (lower bar, same best candidate) ---
                # Reached when dedup found nothing: below the dedup threshold,
                # or the LLM judge explicitly said "not same". Here a close but
                # distinct pair may still be a contradiction.
                if (
                    contra_on
                    and best_sim >= policy.similarity_threshold  # type: ignore[union-attr]
                    and self._llm_available("contradict")
                ):
                    if self._contradicts(
                        dup_text, text, dup_ts, current_ts,
                        policy.show_timestamps,  # type: ignore[union-attr]
                    ):
                        # Newer text (``text``) wins; ``dup_row`` is rewritten,
                        # the row ``text`` came from will be dropped by caller.
                        return dup_row, text, "contradict"
        return None, None, None

    # Public API

    def dedup_items(
        self,
        memory: StructuredMemory,
        items: list[MemoryItem],
    ) -> DedupReport:
        """Dedup a batch of freshly-added ``items`` against a memory's rows.

        For each item, find a duplicate among the memory's *other* rows (rows
        added earlier in this same batch are treated as existing). Confirmed
        duplicates are either merged into the survivor (consolidation on) or
        deleted (consolidation off). Index removals and survivor updates are
        applied incrementally at the end.
        """
        report = DedupReport(checked=len(items))
        if not items:
            return report

        # All rows currently in the table, including the batch just added.
        rows_by_id: dict[int, dict[str, Any]] = {
            int(r["id"]): r for r in memory.all_rows()
        }
        text_col = memory.text_column
        policy = memory.contradiction_policy()

        for item in items:
            row = item.metadata
            row_id = row.get("id")
            if row_id is None:
                continue
            row_id = int(row_id)
            # If this row was already consumed by an earlier merge/delete.
            current = rows_by_id.get(row_id)
            if current is None:
                continue
            user_id = str(current.get("user_id", "default"))
            text = str(current.get(text_col, ""))
            # Compare against all surviving rows; exclude_ids keeps the RAG
            # search from matching this item against itself in the index.
            all_rows = list(rows_by_id.values())
            dup_row, merged_text, reason = self._find_duplicate(
                memory, text, user_id, all_rows,
                exclude_ids={row_id},
                policy=policy,
                current_ts=current.get("created_at"),
            )
            if dup_row is None:
                continue
            dup_id = int(dup_row["id"])
            if merged_text is not None:
                # Rewrite dup_row with merged_text (consolidation) or the newer
                # text (contradiction); in both cases drop the item's row.
                merged = self._merge_rows(dup_row, current, text_col, merged_text)
                memory.update_row(merged)
                memory.delete_row(row_id)
                rows_by_id[dup_id] = merged
                del rows_by_id[row_id]
                if reason == "contradict":
                    report.resolved += 1
                else:
                    report.merged += 1
                report.updated_ids.append(dup_id)
                report.removed_ids.append(row_id)
            else:
                # Drop the newer row (this item); keep the existing one.
                memory.delete_row(row_id)
                del rows_by_id[row_id]
                report.skipped += 1
                report.removed_ids.append(row_id)

        memory.apply_index_changes(
            removed_ids=report.removed_ids,
            updated_ids=report.updated_ids,
        )
        return report

    def sweep(
        self,
        memory: StructuredMemory,
        user_id: Optional[str] = None,
    ) -> DedupReport:
        """Compact a whole memory in place, merging/dropping duplicates.

        When ``user_id`` is given, only that user's rows are swept. When it is
        ``None`` and ``per_user`` is set, each user is swept independently;
        otherwise all rows are compared together. Index deltas are applied once
        at the end.
        """
        cfg = self.config
        report = DedupReport()
        if user_id is not None:
            groups: list[Optional[str]] = [user_id]
        else:
            distinct: list[Optional[str]] = []
            for r in memory.all_rows():
                uid = r.get("user_id")
                if uid not in distinct:
                    distinct.append(uid)
            groups = distinct if cfg.per_user else [None]

        for uid in groups:
            self._sweep_group(memory, uid, report)

        report.checked = sum(len(memory.all_rows(u)) for u in groups)
        memory.apply_index_changes(
            removed_ids=report.removed_ids,
            updated_ids=report.updated_ids,
        )
        return report

    def _sweep_group(
        self,
        memory: StructuredMemory,
        user_id: Optional[str],
        report: DedupReport,
    ) -> None:
        cfg = self.config
        text_col = memory.text_column
        policy = memory.contradiction_policy()
        contra_on = bool(policy.enabled and policy.similarity_threshold is not None)
        rows = memory.all_rows(user_id)
        if len(rows) < 2:
            return

        # Earliest first → survivors are the oldest entries.
        rows.sort(key=lambda r: (float(r.get("created_at") or 0.0), int(r.get("id") or 0)))
        texts = [str(r.get(text_col, "")) for r in rows]
        vecs = self._embed_normalized(texts)

        # survivors holds (row_dict, normalized_vec) for rows kept so far.
        survivors: list[tuple[dict[str, Any], np.ndarray]] = [(rows[0], vecs[0])]

        for i in range(1, len(rows)):
            row = rows[i]
            text = texts[i]
            vec = vecs[i]
            dup_row: Optional[dict[str, Any]] = None
            contra_row: Optional[dict[str, Any]] = None

            # Gate 1: exact against current survivors.
            if cfg.exact:
                norm = _normalize(text)
                for s_row, _ in survivors:
                    if _normalize(str(s_row.get(text_col, ""))) == norm:
                        dup_row = s_row
                        break

            # Cosine against current survivors — shared by dedup + contradiction.
            best_j: Optional[int] = None
            best_sim: float = 0.0
            if dup_row is None and (cfg.similarity_threshold is not None or contra_on):
                surv_mat = np.array([v for _, v in survivors])
                sims = surv_mat @ vec
                best_j = int(np.argmax(sims))
                best_sim = float(sims[best_j])
                s_row = survivors[best_j][0]
                s_text = str(s_row.get(text_col, ""))

                # Gate 2: dedup similarity + Gate 3 (LLM judge). When the judge
                # is off, the candidate is a tentative duplicate and is
                # disambiguated against the contradiction gate below.
                is_dup = False
                if cfg.similarity_threshold is not None and best_sim >= cfg.similarity_threshold:
                    if cfg.llm_judge and self._llm_available("llm_judge"):
                        is_dup = self._judge(text, s_text)
                    else:
                        is_dup = True

                # A tentative duplicate may actually be a contradiction.
                if is_dup and contra_on and self._llm_available("contradict"):
                    if self._contradicts(
                        s_text, text,
                        s_row.get("created_at"), row.get("created_at"),
                        policy.show_timestamps,
                    ):
                        contra_row = s_row
                        is_dup = False

                if is_dup:
                    dup_row = s_row
                elif contra_row is None and contra_on and best_sim >= policy.similarity_threshold \
                        and self._llm_available("contradict"):
                    # Below dedup bar (or judge said not-same): maybe a
                    # contradiction at the lower bar.
                    if self._contradicts(
                        s_text, text,
                        s_row.get("created_at"), row.get("created_at"),
                        policy.show_timestamps,
                    ):
                        contra_row = s_row

            if dup_row is not None:
                # Confirmed duplicate of dup_row (a survivor).
                loser_id = int(row["id"])
                dup_id = int(dup_row["id"])
                if cfg.consolidate and self._llm_available("consolidate"):
                    merged_text = self._consolidate(str(dup_row.get(text_col, "")), text)
                    if merged_text:
                        merged_row = self._merge_rows(dup_row, row, text_col, merged_text)
                        memory.update_row(merged_row)
                        memory.delete_row(loser_id)
                        # Re-embed the survivor so later rows compare to the merged text.
                        new_vec = self._embed_normalized([merged_text])[0]
                        survivors = [
                            (merged_row, new_vec) if s is dup_row else s
                            for s in survivors
                        ]
                        report.merged += 1
                        report.updated_ids.append(dup_id)
                        report.removed_ids.append(loser_id)
                        continue
                # No consolidation (or it failed) — drop the loser.
                memory.delete_row(loser_id)
                report.skipped += 1
                report.removed_ids.append(loser_id)
                continue

            if contra_row is not None:
                # Contradiction: overwrite the older survivor's text with the
                # newer row's text, then drop the newer row.
                loser_id = int(row["id"])
                contra_id = int(contra_row["id"])
                merged_row = self._merge_rows(contra_row, row, text_col, text)
                memory.update_row(merged_row)
                memory.delete_row(loser_id)
                new_vec = self._embed_normalized([text])[0]
                survivors = [
                    (merged_row, new_vec) if s is contra_row else s
                    for s in survivors
                ]
                report.resolved += 1
                report.updated_ids.append(contra_id)
                report.removed_ids.append(loser_id)
                continue

            survivors.append((row, vec))

"""Small literal dotenv reader/writer shared by configuration and setup.

No interpolation, shell evaluation, or multiline values. Double-quoted values
escape only backslashes and double quotes, so paths and literal ``$`` survive.
"""

import re
from pathlib import Path

_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)$")


def assignment(line: str) -> tuple[str, str, str] | None:
    """Return key, literal value, and any trailing quoted-value comment."""
    match = _ASSIGNMENT.match(line.rstrip("\r\n"))
    if not match:
        return None
    key, raw = match.groups()
    raw = raw.strip()
    if raw[:1] in {"'", '"'}:
        quote = raw[0]
        value = []
        pos = 1
        while pos < len(raw):
            char = raw[pos]
            if quote == '"' and char == "\\" and pos + 1 < len(raw) and raw[pos + 1] in {'"', "\\"}:
                pos += 1
                value.append(raw[pos])
            elif char == quote:
                tail = raw[pos + 1:].strip()
                if not tail or tail.startswith("#"):
                    return key, "".join(value), tail
                break
            else:
                value.append(char)
            pos += 1
    # Preserve the legacy literal treatment of unquoted values, including #.
    return key, raw, ""


def read_values(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        item = assignment(line)
        if item:
            values.setdefault(item[0], item[1])
    return values


def read_text(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as stream:
        return stream.read()


def quote_value(value: str) -> str:
    if any(char in value for char in "\r\n\x00\v\f\x1c\x1d\x1e\x85\u2028\u2029"):
        raise ValueError("Environment values must be single-line and contain no NUL.")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def update_text(text: str, updates: dict[str, str | None]) -> str:
    """Update managed keys once; None removes an assignment (restores fallback)."""
    pending = dict(updates)
    lines = []
    for line in text.splitlines(keepends=True):
        item = assignment(line)
        if not item or item[0] not in updates:
            lines.append(line)
            continue
        key, _, comment = item
        if key in pending:
            value = pending.pop(key)
            if value is not None:
                lines.append(f"{key}={quote_value(value)}" + (f" {comment}" if comment else "") + "\n")
                continue
        if comment:
            lines.append(comment + "\n")
    if pending and lines and not lines[-1].endswith(("\n", "\r")):
        lines.append("\n")
    lines.extend(f"{key}={quote_value(value)}\n" for key, value in pending.items() if value is not None)
    return "".join(lines)

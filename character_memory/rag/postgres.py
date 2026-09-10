"""Durable PostgreSQL full-text + pgvector retrieval, with shared RRF scoring."""
import hashlib
import json
import re
import threading
import uuid
import weakref

import numpy as np

from .base import RAGSystem, as_queries
from .fusion import fuse
from ..chunking.base import Chunk
from ..memory.store_base import identifier

_LOCKS = weakref.WeakValueDictionary()
_LOCKS_GUARD = threading.Lock()


class PostgresHybridSearch(RAGSystem):
    name = 'postgres'
    remote = True

    def __init__(self, embedder, store, collection, *, text_search_config='simple',
                 candidate_pool=30, rrf_k=60, min_dense_similarity=None,
                 hnsw=True, ef_search=100, batch_size=256):
        self.embedder, self.store, self.collection = embedder, store, collection
        self.text_search_config = text_search_config
        self.candidate_pool, self.rrf_k = candidate_pool, rrf_k
        self.min_dense_similarity = min_dense_similarity
        self.hnsw, self.ef_search, self.batch_size = hnsw, ef_search, batch_size
        if min_dense_similarity is not None and not -1 <= min_dense_similarity <= 1:
            raise ValueError('min_dense_similarity must be between -1 and 1')
        if batch_size < 1 or ef_search < 1 or candidate_pool < 1 or rrf_k < 0:
            raise ValueError('Invalid retrieval sizing')
        self._refresh_callback = None
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault((store._pool_key, store.schema, collection), threading.RLock())
        with store.connection() as conn:
            if conn.execute("SELECT to_regtype('public.vector') AS t").fetchone()['t'] is None:
                raise RuntimeError('pgvector is missing: install the extension and run CREATE EXTENSION vector in this database')
            conn.execute('SELECT %s::regconfig', [text_search_config])
            conn.execute('''CREATE TABLE IF NOT EXISTS cm_collections (
                name TEXT PRIMARY KEY, active_table TEXT NOT NULL, dimension INTEGER NOT NULL,
                fingerprint TEXT NOT NULL, lexical_config TEXT NOT NULL)''')

    def _fingerprint(self):
        return hashlib.sha256(json.dumps({
            'provider': type(self.embedder).__module__ + '.' + type(self.embedder).__qualname__,
            'identity': getattr(self.embedder, 'index_fingerprint', None),
            'normalization': 'l2-v1',
        }, sort_keys=True).encode()).hexdigest()

    def _state(self):
        rows = self.store.execute('SELECT * FROM cm_collections WHERE name=?', [self.collection])
        return rows[0] if rows else None

    def exists(self, path=''):
        return self._state() is not None

    def _vectors(self, texts, *, query=False):
        method = self.embedder.embed_queries if query else self.embedder.embed_documents
        vectors = np.asarray(method(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
            raise ValueError('Embedding provider returned invalid vectors')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    @staticmethod
    def _vector(vector):
        return '[' + ','.join(map(str, vector.tolist())) + ']'

    def _insert(self, conn, table, chunks, vectors):
        with conn.cursor() as cur:
            cur.executemany(
                f'INSERT INTO {identifier(table)} (text,source,metadata,embedding,lexical) '
                'VALUES (%s,%s,%s::jsonb,%s::vector,to_tsvector(%s::regconfig,%s))',
                [(c.text, c.source, json.dumps({"source": c.source, **c.metadata}), self._vector(v), self.text_search_config, c.text)
                 for c, v in zip(chunks, vectors)],
            )

    def build(self, chunks, *, pending=()):
        """Publish a replacement generation only after embedding and indexing succeed."""
        with self._lock:
            table = 'cm_vectors_' + uuid.uuid4().hex
            dim = int(self.embedder.dim)
            try:
                with self.store.connection() as conn:
                    conn.execute(f'''CREATE TABLE {identifier(table)} (
                        doc_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        text TEXT NOT NULL, source TEXT NOT NULL, metadata JSONB NOT NULL,
                        embedding vector({dim}) NOT NULL, lexical tsvector NOT NULL)''')
                for start in range(0, len(chunks), self.batch_size):
                    batch = chunks[start:start + self.batch_size]
                    vectors = self._vectors([c.text for c in batch])
                    if vectors.shape[1] != dim:
                        raise ValueError('Embedding dimension changed during collection build')
                    with self.store.connection() as conn:
                        self._insert(conn, table, batch, vectors)
                with self.store.connection() as conn:
                    conn.execute(f'CREATE INDEX ON {identifier(table)} USING gin (lexical)')
                    conn.execute(f'CREATE INDEX ON {identifier(table)} USING gin (metadata jsonb_path_ops)')
                    if self.hnsw and dim <= 2000:
                        conn.execute(f'CREATE INDEX ON {identifier(table)} USING hnsw (embedding vector_cosine_ops)')
                    conn.execute(f'ANALYZE {identifier(table)}')
                    old = conn.execute('SELECT active_table FROM cm_collections WHERE name=%s', [self.collection]).fetchone()
                    conn.execute('''INSERT INTO cm_collections VALUES (%s,%s,%s,%s,%s)
                        ON CONFLICT(name) DO UPDATE SET active_table=excluded.active_table,
                        dimension=excluded.dimension,fingerprint=excluded.fingerprint,lexical_config=excluded.lexical_config''',
                        [self.collection, table, dim, self._fingerprint(), self.text_search_config])
                    for work in pending:
                        conn.execute('DELETE FROM cm_index_work WHERE collection=%s AND row_key=%s AND revision=%s',
                                     [self.collection, work['row_key'], work['revision']])
                    if old:
                        conn.execute(f'DROP TABLE {identifier(old["active_table"])}')
            except BaseException:
                with self.store.connection() as conn:
                    conn.execute(f'DROP TABLE IF EXISTS {identifier(table)}')
                raise

    def _ensure(self, dimension=None):
        with self._lock:
            state = self._state()
            dim = int(dimension if dimension is not None else self.embedder.dim)
            if state is None:
                self.build([])
            elif (state['dimension'] != dim or state['fingerprint'] != self._fingerprint()
                  or state['lexical_config'] != self.text_search_config):
                self.build(self.documents)
            return self._state()

    def add_documents(self, chunks):
        if not chunks:
            return
        vectors = self._vectors([c.text for c in chunks])
        with self._lock:
            state = self._ensure(vectors.shape[1])
            with self.store.connection() as conn:
                self._insert(conn, state['active_table'], chunks, vectors)

    def replace_documents(self, ids, chunks, *, pending=()):
        """Atomic row-key replacement and conditional acknowledgement of source work."""
        vectors = self._vectors([c.text for c in chunks]) if chunks else None
        with self._lock:
            state = self._ensure(vectors.shape[1] if vectors is not None else None)
            with self.store.connection() as conn:
                conn.execute(f'DELETE FROM {identifier(state["active_table"])} WHERE metadata->>\'id\'=ANY(%s)', [list(map(str, ids))])
                if chunks:
                    self._insert(conn, state['active_table'], chunks, vectors)
                for work in pending:
                    conn.execute('DELETE FROM cm_index_work WHERE collection=%s AND row_key=%s AND revision=%s',
                                 [self.collection, work['row_key'], work['revision']])

    def delete_documents(self, ids):
        with self._lock:
            state = self._state()
            if not state:
                return 0
            with self.store.connection() as conn:
                return conn.execute(f'DELETE FROM {identifier(state["active_table"])} WHERE metadata->>\'id\'=ANY(%s)', [list(map(str, ids))]).rowcount

    def cleanup(self):
        return 0  # Physical deletion; PostgreSQL autovacuum reclaims space.

    @property
    def documents(self):
        with self._lock:
            state = self._state()
            if not state:
                return []
            rows = self.store.execute(f'SELECT text,source,metadata FROM {identifier(state["active_table"])} ORDER BY doc_id')
            return [Chunk(text=r['text'], source=r['source'], metadata=r['metadata']) for r in rows]

    @property
    def count(self):
        with self._lock:
            state = self._state()
            return self.store.execute(f'SELECT count(*) AS n FROM {identifier(state["active_table"])}')[0]['n'] if state else 0

    def persist(self, path):
        self.refresh()

    def load(self, path):
        self._ensure()
        self.refresh()

    def refresh(self):
        return bool(self._refresh_callback and self._refresh_callback())

    def search(self, query, k=5, where=None, allowed_ids=None, *, exact=False):
        queries = [(q, w) for q, w in as_queries(query) if w > 0]
        if k <= 0 or not queries:
            return []
        self.refresh()
        vectors = self._vectors([q for q, _ in queries], query=True)
        documents, candidates = {}, []
        with self._lock:
            state = self._ensure(vectors.shape[1])
            table = identifier(state['active_table'])
            filters, params = ['metadata @> %s::jsonb'], [json.dumps({key: value for key, value in (where or {}).items() if value is not None})]
            for field, value in (where or {}).items():
                if value is None:
                    filters.append("(metadata->%s IS NULL OR metadata->%s='null'::jsonb)")
                    params.extend([field, field])
            if allowed_ids is not None:
                ids = [allowed_ids] if isinstance(allowed_ids, (str, bytes)) else list(allowed_ids)
                if not ids:
                    return []
                filters.append("metadata->>'id'=ANY(%s)")
                params.append(list(map(str, ids)))
            condition = ' AND '.join(filters)
            pool = max(self.candidate_pool, k * 3)
            def key(doc_id):
                group = documents[doc_id]['metadata'].get('_result_id')
                return ('group', group) if group is not None else ('doc', doc_id)
            def ranked(rows, dense=False):
                result, seen = [], set()
                for row in rows:
                    doc_id = row['doc_id']
                    documents[doc_id] = row
                    group = key(doc_id)
                    score = float(row['rank_score'])
                    if not np.isfinite(score):
                        score = 0.0
                    if group in seen or (dense and self.min_dense_similarity is not None and score <= self.min_dense_similarity):
                        continue
                    seen.add(group)
                    result.append((doc_id, len(result), score))
                return result
            with self.store.connection() as conn:
                conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", [str(self.ef_search)])
                conn.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
                if exact:
                    conn.execute('SET LOCAL enable_indexscan = off')
                    conn.execute('SET LOCAL enable_bitmapscan = off')
                for (text, _), vector in zip(queries, vectors):
                    v = self._vector(vector)
                    # Conversational keyword retrieval matches any query term,
                    # as BM25 does; plainto_tsquery's AND would discard most
                    # useful lexical evidence in full-sentence user questions.
                    lexical_query = ' OR '.join('"' + token + '"' for token in re.findall(r"\w+", text))
                    lexical = conn.execute(f'''SELECT doc_id,text,source,metadata, ts_rank_cd(lexical,websearch_to_tsquery(%s::regconfig,%s)) AS rank_score
                        FROM {table} WHERE {condition} AND lexical @@ websearch_to_tsquery(%s::regconfig,%s)
                        ORDER BY rank_score DESC,doc_id LIMIT %s''',
                        [self.text_search_config, lexical_query, *params, self.text_search_config, lexical_query, pool]).fetchall()
                    dense = conn.execute(f'''SELECT doc_id,text,source,metadata, 1-(embedding <=> %s::vector) AS rank_score
                        FROM {table} WHERE {condition} ORDER BY embedding <=> %s::vector LIMIT %s''',
                        [v, *params, v, pool]).fetchall()
                    # Very selective HNSW filtering may exhaust its scan budget.
                    # Complete such candidate sets using exact filtered search.
                    if not exact and len(dense) < pool:
                        dense = conn.execute(f'''WITH filtered AS MATERIALIZED (SELECT * FROM {table} WHERE {condition})
                            SELECT doc_id,text,source,metadata,1-(embedding <=> %s::vector) AS rank_score FROM filtered
                            ORDER BY embedding <=> %s::vector,doc_id LIMIT %s''', [*params, v, v, pool]).fetchall()
                    candidates.append((ranked(lexical), ranked(dense, True)))
        hits = fuse(queries, candidates, k, self.rrf_k, self.min_dense_similarity, key,
                    lambda doc_id: (documents[doc_id]['text'], documents[doc_id]['source'], documents[doc_id]['metadata']))
        for hit in hits:
            hit.metadata['lexical_score'] = hit.metadata['bm25_score']
            hit.metadata['lexical_backend'] = 'postgres_full_text'
        return hits

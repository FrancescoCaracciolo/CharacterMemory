"""Transactional reconciliation storage; indexes are derived after commit."""
import json
from .store_base import identifier


def ensure_tables(store):
    store.create_table('memory_revisions', {
        'id': 'INTEGER PRIMARY KEY AUTOINCREMENT', 'memory': 'TEXT NOT NULL',
        'row_id': 'INTEGER NOT NULL', 'before_json': 'TEXT NOT NULL',
        'after_json': 'TEXT NOT NULL', 'evidence_json': 'TEXT NOT NULL',
        'reason': 'TEXT NOT NULL', 'model': 'TEXT NOT NULL', 'applied_at': 'REAL NOT NULL',
    })
    store.execute('CREATE INDEX IF NOT EXISTS cm_revision_lookup ON memory_revisions(memory,row_id,id)')
    store.create_table('memory_reconciliation_pending', {
        'id': 'INTEGER PRIMARY KEY AUTOINCREMENT', 'memory': 'TEXT NOT NULL',
        'removed_id': 'INTEGER NOT NULL', 'updated_id': 'INTEGER',
    })
    store.execute('CREATE INDEX IF NOT EXISTS cm_reconciliation_lookup ON memory_reconciliation_pending(memory,id)')


def history(memory, row_id):
    if not memory.store.columns('memory_revisions'):
        return []
    rows = memory.store.select('memory_revisions', {'memory': memory.name, 'row_id': row_id}, order_by='id')
    return [dict(id=r['id'], reason=r['reason'], model=r['model'], applied_at=r['applied_at'],
                 before=json.loads(r['before_json']), after=json.loads(r['after_json']),
                 evidence=json.loads(r['evidence_json'])) for r in rows]


def pending(memory):
    if not memory.store.columns('memory_reconciliation_pending'):
        return []
    return memory.store.select('memory_reconciliation_pending', {'memory': memory.name}, order_by='id')


def apply(memory, existing, incoming, replacement, evidence, reason, model):
    """Compare snapshots and atomically archive/update/delete. False means stale."""
    from .base import MemoryScope
    if existing['id'] == incoming['id']:
        raise ValueError('Cannot reconcile a row with itself')
    if memory.scope == MemoryScope.PER_USER and existing.get('user_id') != incoming.get('user_id'):
        raise ValueError('Cannot reconcile different user scopes')
    if replacement is not None and replacement['id'] != existing['id']:
        raise ValueError('A revision must preserve the canonical row ID')
    if not getattr(memory, '_reconciliation_tables_ready', False):
        ensure_tables(memory.store)
        memory._reconciliation_tables_ready = True
    table = identifier(memory.table)
    dump = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    with memory.store.transaction(immediate=True) as tx:
        # PostgreSQL also needs row locks against writers not taking the advisory lock.
        suffix = ' FOR UPDATE' if memory.store.durable_indexes else ''
        for snapshot in sorted((existing, incoming), key=lambda r: r['id']):
            row = tx.execute(f'SELECT * FROM {table} WHERE id=?' + suffix, [snapshot['id']]).fetchone()
            if row is None or dict(row) != snapshot:
                return False
        if replacement is not None:
            tx.execute('INSERT INTO memory_revisions (memory,row_id,before_json,after_json,evidence_json,reason,model,applied_at) VALUES (?,?,?,?,?,?,?,?)',
                       [memory.name, existing['id'], dump(existing), dump(replacement), dump(evidence), reason, model, memory._now()])
            cols = [c for c in replacement if c != 'id']
            tx.execute(f'UPDATE {table} SET ' + ','.join(f'{identifier(c)}=?' for c in cols) + ' WHERE id=?',
                       [replacement[c] for c in cols] + [existing['id']])
        tx.execute(f'DELETE FROM {table} WHERE id=?', [incoming['id']])
        tx.execute('INSERT INTO memory_reconciliation_pending (memory,removed_id,updated_id) VALUES (?,?,?)',
                   [memory.name, incoming['id'], existing['id']])
    return True

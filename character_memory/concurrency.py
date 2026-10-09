"""In-process coordination; locks never span database connections and model calls."""
from functools import wraps
import threading
import weakref

_GUARD = threading.Lock()


def object_lock(obj, key='state', *, reentrant=True):
    with _GUARD:
        locks = getattr(obj, "_cm_operation_locks", None)
        if locks is None:
            locks = weakref.WeakValueDictionary()
            setattr(obj, "_cm_operation_locks", locks)
        return locks.setdefault(key, threading.RLock() if reentrant else threading.Lock())


def synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with object_lock(self):
            return method(self, *args, **kwargs)
    return call


class Coalescer:
    """Collapse concurrent calls of one idempotent operation into few runs.

    ``run(fn)`` returns once a run of ``fn`` that *started after* the call has
    completed, so its effects are covered. Callers arriving while a run is in
    flight share the next single run instead of queueing one each. If a run
    fails, the exception goes to the thread that ran it and every waiter it
    would have covered retries. A call from inside ``fn`` (same thread) runs
    inline.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._requested = 0
        self._completed = 0
        self._runner = None

    def run(self, fn):
        me = threading.get_ident()
        with self._cond:
            if self._runner == me:
                reentrant = True
            else:
                reentrant = False
                self._requested += 1
                ticket = self._requested
                while self._completed < ticket and self._runner is not None:
                    self._cond.wait()
                if self._completed >= ticket:
                    return
                self._runner = me
                covers = self._requested
        if reentrant:
            fn()
            return
        try:
            fn()
        except BaseException:
            with self._cond:
                self._runner = None
                self._cond.notify_all()
            raise
        with self._cond:
            self._completed = max(self._completed, covers)
            self._runner = None
            self._cond.notify_all()


def chat_lock(store, chat_id):
    """The non-reentrant lock serializing generation/extraction of one chat."""
    return object_lock(store, ('chat', chat_id), reentrant=False)


def chat_serialized(method):
    """Streaming acquires the lock on iteration and releases it on close/error."""
    @wraps(method)
    def call(self, target, *args, **kwargs):
        chat = self._as_chat(target) if isinstance(target, str) else target
        if not hasattr(chat, 'id'):
            return method(self, target, *args, **kwargs)
        lock = chat_lock(self.store, chat.id)
        if kwargs.get('stream', False):
            def stream():
                with lock:
                    yield from method(self, target, *args, **kwargs)
            return stream()
        with lock:
            return method(self, target, *args, **kwargs)
    return call

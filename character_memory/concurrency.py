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


def chat_serialized(method):
    """Streaming acquires the lock on iteration and releases it on close/error."""
    @wraps(method)
    def call(self, target, *args, **kwargs):
        chat = self._as_chat(target) if isinstance(target, str) else target
        if not hasattr(chat, 'id'):
            return method(self, target, *args, **kwargs)
        lock = object_lock(self.store, ('chat', chat.id), reentrant=False)
        if kwargs.get('stream', False):
            def stream():
                with lock:
                    yield from method(self, target, *args, **kwargs)
            return stream()
        with lock:
            return method(self, target, *args, **kwargs)
    return call

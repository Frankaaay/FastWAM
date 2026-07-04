import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass
class _PrefetchError:
    error: BaseException


_STOP = object()


class AsyncBatchPrefetcher(Iterator[T], Generic[T]):
    def __init__(
        self,
        iterable: Iterable[T],
        *,
        max_prefetch: int,
        prepare_fn: Callable[[T], T] | None = None,
        name: str = "async-batch-prefetch",
    ):
        max_prefetch = int(max_prefetch)
        if max_prefetch <= 0:
            raise ValueError(f"`max_prefetch` must be positive, got {max_prefetch}.")
        self._iterator = iter(iterable)
        self._prepare_fn = prepare_fn
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def __iter__(self):
        return self

    def __next__(self) -> T:
        item = self._queue.get()
        if item is _STOP:
            self.close()
            raise StopIteration
        if isinstance(item, _PrefetchError):
            self.close()
            raise item.error
        return item  # type: ignore[return-value]

    def _put(self, item: object) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            for item in self._iterator:
                if self._stop.is_set():
                    break
                if self._prepare_fn is not None:
                    item = self._prepare_fn(item)
                if not self._put(item):
                    break
        except BaseException as exc:
            self._put(_PrefetchError(exc))
        finally:
            self._put(_STOP)

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

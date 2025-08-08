from __future__ import annotations
from collections import deque
from typing import Deque, Iterable, List


class SMA:
    def __init__(self, window: int) -> None:
        if window <= 0:
            raise ValueError("window must be > 0")
        self.window = window
        self._buf: Deque[float] = deque(maxlen=window)
        self.value: float | None = None

    def update(self, price: float) -> float | None:
        self._buf.append(price)
        if len(self._buf) < self.window:
            self.value = None
        else:
            self.value = sum(self._buf) / self.window
        return self.value

    @staticmethod
    def batch(prices: Iterable[float], window: int) -> List[float | None]:
        sma = SMA(window)
        out: List[float | None] = []
        for p in prices:
            out.append(sma.update(p))
        return out 
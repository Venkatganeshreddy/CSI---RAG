"""Token-bucket rate limiter for Google Sheets API (55 req / 60s)."""

import time
from config.settings import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS


class RateLimiter:
    def __init__(
        self,
        max_requests: int = RATE_LIMIT_REQUESTS,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
    ):
        self._max = max_requests
        self._window = window_seconds
        self._timestamps: list[float] = []

    def acquire(self) -> None:
        """Block until a request slot is available."""
        now = time.monotonic()
        # Purge timestamps outside the window
        self._timestamps = [t for t in self._timestamps if now - t < self._window]

        if len(self._timestamps) >= self._max:
            sleep_for = self._timestamps[0] + self._window - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            # Purge again after sleeping
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < self._window]

        self._timestamps.append(time.monotonic())

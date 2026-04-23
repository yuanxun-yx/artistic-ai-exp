import asyncio


class Budget:
    def __init__(self, max_calls: int):
        if max_calls <= 0:
            raise ValueError("max_calls should be > 0")
        self._remaining = max_calls
        self._lock = asyncio.Lock()

    async def consume(self):
        async with self._lock:
            if self._remaining <= 0:
                raise RuntimeError("Budget exhausted")
            self._remaining -= 1

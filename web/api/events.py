"""Per-job event log on Redis Streams.

Each job has one stream, ``job:{id}:events``. Entries are written with the explicit ID
``0-<seq>``, so the stream is totally ordered by ``seq`` and Redis itself rejects a
non-increasing ID: if two workers ever ran the same job, the second one fails instead of
writing duplicates. Clients replay from any ``seq`` (SSE ``Last-Event-ID``) without loss.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from redis.asyncio import Redis

from .schemas import TERMINAL_EVENTS

Event = dict[str, Any]
HEARTBEAT: Event = {"type": "_heartbeat"}


def stream_key(job_id: str) -> str:
    return f"job:{job_id}:events"


def lock_key(job_id: str) -> str:
    return f"job:{job_id}:lock"


def _seq_of(entry_id: bytes | str) -> int:
    raw = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
    return int(raw.split("-")[1])


def _decode(fields: dict[Any, Any]) -> Event:
    payload = fields.get(b"e", fields.get("e"))
    if payload is None:
        raise ValueError("Malformed event entry (no 'e' field)")
    event: Event = json.loads(payload)
    return event


class EventLog:
    """Append-only event log of one job."""

    def __init__(self, redis: Redis, job_id: str, ttl_seconds: int = 24 * 3600) -> None:
        self.redis = redis
        self.job_id = job_id
        self.key = stream_key(job_id)
        self.ttl = ttl_seconds

    async def last_seq(self) -> int:
        """Highest ``seq`` written so far (0 if none)."""
        last = await self.redis.xrevrange(self.key, count=1)
        return _seq_of(last[0][0]) if last else 0

    async def publish(self, event: Event, seq: int) -> Event:
        """Append ``event`` with the given ``seq`` and refresh the TTL.

        Raises:
            redis.ResponseError: If ``seq`` is not greater than the last one written.
        """
        stored = {**event, "seq": seq, "jobId": self.job_id}
        await self.redis.xadd(self.key, {"e": json.dumps(stored)}, id=f"0-{seq}")
        await self.redis.expire(self.key, self.ttl)
        return stored

    async def replay(self, after_seq: int = 0) -> list[Event]:
        """All events with ``seq > after_seq``, in order."""
        rows = await self.redis.xrange(self.key, min=f"(0-{after_seq}", max="+")
        return [_decode(fields) for _, fields in rows]

    async def tail(self, after_seq: int, block_ms: int = 15_000) -> AsyncIterator[Event]:
        """Yield events after ``after_seq``, then wait for new ones.

        Yields :data:`HEARTBEAT` every ``block_ms`` without events (keeps proxies from
        closing the connection) and stops after a terminal event.
        """
        last_id = f"0-{after_seq}"
        while True:
            resp = await self.redis.xread({self.key: last_id}, block=block_ms, count=200)
            if not resp:
                yield HEARTBEAT
                continue
            for entry_id, fields in resp[0][1]:
                last_id = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                event = _decode(fields)
                yield event
                if event["type"] in TERMINAL_EVENTS:
                    return

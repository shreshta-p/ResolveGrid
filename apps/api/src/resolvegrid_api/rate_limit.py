"""In-memory, single-process rate limiter.

Deliberately not Redis-backed: this process is the only writer today (no
horizontal scaling, no multi-worker deployment yet per the modular-monolith
ADR), so an in-memory sliding window is sufficient and avoids introducing a
Redis dependency into apps/api before anything actually needs it. Revisit
(Redis-backed, shared across workers) once apps/api runs as more than one
process.
"""

import time
from collections import defaultdict

from fastapi import Depends, HTTPException

from resolvegrid_api.deps import get_principal
from resolvegrid_authz import Principal

_WINDOW_SECONDS = 60
_MAX_REQUESTS_PER_WINDOW = 10
_request_log: dict[int, list[float]] = defaultdict(list)


def check_ticket_creation_rate_limit(principal: Principal = Depends(get_principal)) -> Principal:
    now = time.monotonic()
    log = _request_log[principal.employee_id]
    log[:] = [t for t in log if now - t < _WINDOW_SECONDS]
    if len(log) >= _MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Too many tickets created recently, please wait")
    log.append(now)
    return principal

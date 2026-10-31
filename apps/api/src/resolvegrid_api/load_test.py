"""Phase 14 Task 1: load-test harness core.

Module location: `resolvegrid_api.load_test`, not `scripts/load_test.py`
------------------------------------------------------------------------
The phase plan doc (`docs/superpowers/plans/2026-10-29-phase14-
performance-load-testing.md`, Task 1) flagged this as an open decision,
to be resolved empirically by which import path is cleaner given this
module's real DB needs. Chosen: inside the `resolvegrid_api` package,
mirroring `eval_retrieval.py`/`diagnostic_eval.py`'s established
precedent. Concretely: `build_workload_mix` below needs a real `Session`
to draw real seeded `Employee`/`Queue`/`Ticket` rows (not synthetic
placeholder ids), and Task 2 (the next task in this phase) will reuse
this exact session/engine machinery again to write `LoadTestRun`
manifest rows and to query real `AgentRun`/`Span`/`ModelCall` telemetry
for latency/cost metrics. Putting this in `scripts/` would mean either
importing the whole `resolvegrid_api` package from outside it (undoing
the layering `eval_retrieval.py` already settled) or duplicating DB
wiring -- neither is cleaner. A thin CLI entry point (`main()`) at the
bottom still makes this runnable exactly like `eval_retrieval.py`'s own
`python -m resolvegrid_api.load_test`.

NHPP arrival sampling via Lewis-Shedler thinning
------------------------------------------------------------------------
"Realistic arrivals/bursts/concurrency" means NOT a constant interarrival
rate. Arrivals here are modeled as a non-homogeneous Poisson process
(NHPP) with a real, day-shaped intensity curve `intensity(t, config)`
(see below), sampled via **thinning**: generate candidate arrivals from a
homogeneous Poisson process at the curve's own maximum rate
(`lambda_max`, found by grid search over one 24h period), then accept
each candidate independently with probability `intensity(t)/lambda_max`.
This is the standard, citable technique for simulating an NHPP from a
bounded intensity function -- Lewis, P.A.W. and Shedler, G.S. (1979),
"Simulation of nonhomogeneous Poisson processes by thinning", Naval
Research Logistics Quarterly 26(3), 403-413 -- not invented arrival
logic. Implemented with only the stdlib `random`/`math` modules, per the
plan doc's explicit choice not to add a new runtime dependency for this.

Day-shaped intensity curve
------------------------------------------------------------------------
`intensity()` is `base_rate_rps` (the overnight trough) plus a Gaussian
bump centered at each hour in `peak_hours` (default two: mid-morning and
early-afternoon), each bump rising to `peak_rate_rps` at its center and
decaying with standard deviation `peak_width_hours`. This shape (two
business-hours peaks separated by a midday dip, falling to a quiet
overnight trough) is a commonly observed pattern in IT-service-desk
ticket volume -- a first wave as the day starts and overnight issues get
reported, a second wave after lunch, then tapering off overnight -- not
a constant or uniformly random rate, which is this task's explicit
requirement. The curve is evaluated on `t mod 24h` (circular distance to
each peak hour), so it repeats identically every 24h of simulated
duration regardless of how long a given run is configured for.

Burst injection
------------------------------------------------------------------------
Independent of the NHPP component above: `burst_schedule` names a few
fixed times (seconds from the run's start) at which a *separate* Poisson
sub-process fires -- `burst_mean_size` simultaneous-ish arrivals (each
jittered within `burst_jitter_seconds` of the scheduled time, so they
land close together without being literally identical), marked
`is_burst=True` so downstream reporting (Task 2) can separate "did the
system survive its sustained day-shaped load" from "did it survive a
sudden concentrated spike" -- exactly the two real failure modes
`PLAN_APPROVED.md` §12's rate-limiting-under-burst requirement cares
about.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api.models import Employee, LoadTestRun, Queue, RoleAssignment, Ticket

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Real seeded data sources (see module-level "realistic workload mix"
# section below for why these specific files).
_CHAT_GOLDEN_PATH = _REPO_ROOT / "eval" / "golden" / "v2" / "chat_v1.jsonl"
_TICKET_TEXT_GOLDEN_PATH = _REPO_ROOT / "eval" / "golden" / "diagnostic_v1.jsonl"

SECONDS_PER_DAY = 24 * 3600.0


# ---------------------------------------------------------------------------
# NHPP arrival sampler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arrival:
    """One sampled arrival: `t` is seconds since the run's start (0 <=
    t < duration_seconds). `is_burst` distinguishes a scheduled burst
    sub-process arrival from an ordinary day-shaped-curve arrival -- see
    module docstring's "Burst injection" section.
    """

    t: float
    is_burst: bool


@dataclass(frozen=True)
class NHPPConfig:
    """Parameters for the day-shaped NHPP intensity curve plus burst
    injection. See module docstring for the full rationale of each shape
    choice. `base_rate_rps`/`peak_rate_rps` are the primary, explicit
    knobs (directly testable); `mean_intensity_rps()` below derives the
    resulting real time-averaged rate (the "target rps" the phase plan's
    `LoadTestRun.config_json` schema note calls for) empirically from
    them, rather than the caller having to pre-compute it by hand.
    """

    duration_seconds: float
    base_rate_rps: float
    peak_rate_rps: float
    peak_hours: tuple[float, ...] = (10.0, 14.5)
    peak_width_hours: float = 1.5
    burst_schedule: tuple[float, ...] = ()
    burst_mean_size: float = 15.0
    burst_jitter_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.base_rate_rps < 0:
            raise ValueError("base_rate_rps must be non-negative")
        if self.peak_rate_rps < self.base_rate_rps:
            raise ValueError("peak_rate_rps must be >= base_rate_rps")
        if self.peak_width_hours <= 0:
            raise ValueError("peak_width_hours must be positive")
        if self.burst_mean_size < 0:
            raise ValueError("burst_mean_size must be non-negative")


def _circular_distance(a: float, b: float, period: float) -> float:
    """Shortest distance between `a` and `b` on a circle of length
    `period` -- used so a peak hour near midnight (e.g. 23.5) still
    attracts arrivals just after 0.0, rather than being treated as far
    away on a naive linear scale.
    """
    d = abs(a - b) % period
    return min(d, period - d)


def intensity(t_seconds: float, config: NHPPConfig) -> float:
    """Real arrivals/second at time `t_seconds` (seconds since run
    start) under `config`'s day-shaped curve. See module docstring.
    """
    hour_of_day = (t_seconds / 3600.0) % 24.0
    value = config.base_rate_rps
    for peak_hour in config.peak_hours:
        distance_hours = _circular_distance(hour_of_day, peak_hour, 24.0)
        value += (config.peak_rate_rps - config.base_rate_rps) * math.exp(
            -0.5 * (distance_hours / config.peak_width_hours) ** 2
        )
    return value


def _intensity_grid(config: NHPPConfig, resolution: int = 2000) -> list[float]:
    """`intensity()` sampled at `resolution` evenly spaced points across
    one 24h period (resolution=2000 -> ~43.2s steps -- far finer than
    this harness's second-scale duration/rps parameters need). Shared by
    `_compute_lambda_max` (thinning's required upper bound) and
    `mean_intensity_rps` (the empirical "target rps") so both are
    consistent with the exact same discretization of the same curve.
    """
    return [intensity(i * SECONDS_PER_DAY / resolution, config) for i in range(resolution)]


def _compute_lambda_max(config: NHPPConfig) -> float:
    """Upper bound on `intensity()` over the whole curve, for thinning's
    candidate-generation rate. A 0.1% safety margin is added on top of
    the grid's observed max: the grid is a finite sample of a continuous
    function, so the true maximum (at a peak_hour's exact center) could
    fall slightly between grid points; thinning's correctness requires
    `lambda_max >= sup intensity(t)` (never merely close to it), and this
    curve's true maximum is analytically known to occur exactly at each
    `peak_hour` (value = peak_rate_rps, when peaks don't overlap) or
    higher (if two peaks are close enough to add constructively) -- the
    margin absorbs that discretization gap cheaply without materially
    slowing thinning's acceptance rate.
    """
    return max(_intensity_grid(config)) * 1.001


def mean_intensity_rps(config: NHPPConfig) -> float:
    """The real time-averaged arrival rate implied by `config`, computed
    empirically from the same grid `_compute_lambda_max` uses -- this is
    the actual "target rps" value recorded into `LoadTestRun.config_json`
    (see `build_run_config_json`), derived from the shape parameters
    rather than asked for as a separate, potentially-inconsistent input.
    """
    grid = _intensity_grid(config)
    return sum(grid) / len(grid)


def _sample_poisson(rng: random.Random, mean: float) -> int:
    """Samples one draw from a Poisson(`mean`) distribution using only
    uniform random draws from `rng` -- Knuth's algorithm (Knuth, D.E.,
    *The Art of Computer Programming, Vol. 2: Seminumerical Algorithms*,
    3rd ed., 1997, Section 3.4.1, "Poisson random variables"). Used instead of
    any extra dependency because `random.Random` has no built-in Poisson
    variate, and because routing every draw (including burst sizes)
    through the same seeded `rng` is what makes `sample_nhpp_arrivals`
    fully deterministic given a fixed seed.
    """
    if mean <= 0:
        return 0
    limit = math.exp(-mean)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= limit:
            return k - 1


def sample_nhpp_arrivals(config: NHPPConfig, seed: int) -> list[Arrival]:
    """Samples a full, deterministic arrival sequence for `config`,
    given `seed`. Two independent components, both documented in the
    module docstring:

    1. The day-shaped NHPP, via Lewis-Shedler thinning: homogeneous
       Poisson candidates at `lambda_max`, each accepted with
       probability `intensity(t)/lambda_max`.
    2. Burst injection at each `config.burst_schedule` time: a
       Poisson(`burst_mean_size`) number of arrivals, each jittered
       within `burst_jitter_seconds`, marked `is_burst=True`.

    The returned list is sorted by `t` (burst arrivals are generated
    after the full day-shaped pass, so sorting is required to produce a
    single, time-ordered schedule). Fully deterministic: a single
    `random.Random(seed)` instance drives every draw in a fixed order, so
    the same `(config, seed)` always yields byte-identical output --
    this is the property `test_load_test.py`'s required correctness test
    checks directly.
    """
    rng = random.Random(seed)
    lambda_max = _compute_lambda_max(config)

    arrivals: list[Arrival] = []
    t = 0.0
    while True:
        t += rng.expovariate(lambda_max)
        if t >= config.duration_seconds:
            break
        if rng.random() <= intensity(t, config) / lambda_max:
            arrivals.append(Arrival(t=t, is_burst=False))

    for burst_time in config.burst_schedule:
        if not (0.0 <= burst_time < config.duration_seconds):
            continue
        count = _sample_poisson(rng, config.burst_mean_size)
        for _ in range(count):
            jitter = rng.uniform(0.0, config.burst_jitter_seconds)
            t_arrival = min(burst_time + jitter, config.duration_seconds - 1e-6)
            arrivals.append(Arrival(t=t_arrival, is_burst=True))

    arrivals.sort(key=lambda a: a.t)
    return arrivals


# ---------------------------------------------------------------------------
# Realistic workload-mix generator
# ---------------------------------------------------------------------------

# Documented assumption (plan doc Task 2 requires this be "stated
# explicitly"): no real pre-launch usage telemetry exists yet to measure
# an actual endpoint mix from, so this is a reasoned default, not a
# measured fact -- /chat (general KB Q&A) is assumed the highest-volume,
# lowest-cost path; ticket creation is the core ITSM action; the
# diagnostic specialist (/tickets/{id}/diagnose) is the most
# model-call-expensive path (ADR 0003/0004's own real findings) and is
# assumed invoked on a minority of tickets that need deeper
# investigation, not every ticket. Task 2 may revisit this ratio once
# real measured data exists.
DEFAULT_WORKLOAD_MIX: dict[str, float] = {
    "chat": 0.50,
    "create_ticket": 0.35,
    "diagnose": 0.15,
}


@dataclass(frozen=True)
class WorkloadItem:
    """One request this harness will fire: which real endpoint, acting
    as which real employee, with what payload/path parameters. `kind` is
    one of "chat" | "create_ticket" | "diagnose" -- see `_METHOD_PATH`.
    """

    kind: str
    employee_id: int
    payload: dict | None = None
    path_params: dict = field(default_factory=dict)


def _load_jsonl_field(path: Path, field_name: str) -> list[str]:
    values: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values.append(json.loads(line)[field_name])
    if not values:
        raise ValueError(f"{path} contained no usable {field_name!r} values")
    return values


def load_real_chat_questions(path: Path | None = None) -> list[str]:
    """Real, hand-written `/chat` questions from `eval/golden/v2/
    chat_v1.jsonl`'s `input_text` field -- the same golden set Phase 10's
    chat-dimension eval already uses, reused here rather than inventing
    new synthetic question strings, per this task's explicit instruction.
    """
    return _load_jsonl_field(path or _CHAT_GOLDEN_PATH, "input_text")


def load_real_ticket_descriptions(path: Path | None = None) -> list[str]:
    """Real, hand-written ticket subject/body text from `eval/golden/
    diagnostic_v1.jsonl`'s `input_text` field -- each is a realistic
    ticket-opening description (e.g. "VPN connection fails after password
    reset") already used as real ticket fixture content by Phase 13's
    `diagnostic_eval.py`, reused here for the same reason as
    `load_real_chat_questions` above: real seeded content, not invented
    placeholder strings.
    """
    return _load_jsonl_field(path or _TICKET_TEXT_GOLDEN_PATH, "input_text")


def fetch_real_employee_ids(session: Session) -> list[int]:
    """Real, currently-active `Employee.id` values from whatever org data
    is actually seeded in `session`'s DB (e.g. via `resolvegrid_api.seed.
    generate_org`) -- never a synthetic/placeholder id range.
    """
    ids = list(session.scalars(select(Employee.id).where(Employee.employment_status == "active")).all())
    if not ids:
        raise ValueError(
            "no active Employee rows found -- seed the org directory first "
            "(resolvegrid_api.seed.generate_org / scripts/seed_employees.sh)"
        )
    return ids


def fetch_real_queue_ids(session: Session) -> list[int]:
    """Real `Queue.id` values currently in the DB. Raises if none exist
    (rather than fabricating one) -- a `create_ticket` workload item
    needs a real `queue_id` to be valid against `TicketCreateRequest`."""
    ids = list(session.scalars(select(Queue.id)).all())
    if not ids:
        raise ValueError("no Queue rows found -- at least one real queue must exist before building a workload mix that includes create_ticket")
    return ids


def fetch_real_diagnose_capable_employee_ids(session: Session) -> list[int]:
    """Real `Employee.id` values that can actually pass `packages/authz`'s
    `authorize(principal, "ticket.diagnose")` check -- a real gap found
    while running this harness against the live stack (Task 2): `ticket.
    diagnose` is in `_STAFF_ONLY_ACTIONS` (`policy.py`), denied outright
    for a principal with no global-admin or department-scoped analyst/
    approver grant. The original `build_workload_mix` drew EVERY kind's
    `employee_id` (including "diagnose") from `fetch_real_employee_ids`'s
    full active-employee pool -- realistic for "chat"/"create_ticket"
    (any employee may ask a question or file a ticket) but not for
    "diagnose" (only IT staff run ticket diagnosis in this product), and
    in practice meant ~90% of sampled "diagnose" calls would 403 against
    this seeded org (6 of 58 active employees hold any role grant).
    Restricted here to global admins (`role="admin", scope="global"`,
    which `_is_global_admin`/`authorize()` let diagnose ANY ticket
    regardless of department) -- the smallest real pool that reliably
    reflects actual product usage without depending on which specific
    ticket's queue/department a department-scoped analyst grant happens
    to cover. Raises if none exist, exactly like the other `fetch_real_*`
    helpers: a load test must never fabricate a fake staff id.
    """
    ids = list(
        session.scalars(
            select(RoleAssignment.employee_id).where(
                RoleAssignment.role == "admin", RoleAssignment.scope == "global"
            )
        ).all()
    )
    if not ids:
        raise ValueError(
            "no global-admin RoleAssignment rows found -- a load test including "
            "'diagnose' needs at least one real employee who can pass the "
            "ticket.diagnose staff-only authz check"
        )
    return sorted(set(ids))


def fetch_real_ticket_ids(session: Session) -> list[int]:
    """Real `Ticket.id` values currently in the DB, if any. Empty is a
    valid, expected answer in a freshly seeded dev DB (tickets only exist
    once some have actually been created) -- `build_workload_mix` handles
    that case explicitly rather than failing.
    """
    return list(session.scalars(select(Ticket.id)).all())


def build_workload_mix(
    session: Session,
    rng: random.Random,
    size: int,
    *,
    mix: dict[str, float] = DEFAULT_WORKLOAD_MIX,
) -> list[WorkloadItem]:
    """Builds `size` `WorkloadItem`s, each drawing real seeded employee
    ids, real seeded queue ids, and real hand-written question/ticket
    text -- never synthetic placeholder strings. `rng` (a caller-supplied
    `random.Random`, not the global `random` module) makes this
    deterministic given a fixed seed, exactly like `sample_nhpp_arrivals`.

    If no real `Ticket` rows exist yet in this DB (a plausible state for
    a freshly seeded dev environment -- tickets only exist once some have
    actually been created), the "diagnose" share of `mix` is folded into
    "create_ticket" instead of raising: this is a documented, honest
    fallback (growing the ticket pool for a future run), not a silent
    fabrication of a fake ticket id.

    Only fetches the real data sources a kind actually present in `mix`
    needs: a caller passing e.g. `mix={"chat": 1.0}` must never have to
    have a real `Queue` row seeded, and must never trip `fetch_real_
    queue_ids`'s "no Queue rows found" error, for a workload that was
    never going to create a ticket in the first place -- fetching
    unconditionally would make this function fail on an irrelevant
    precondition instead of the one it actually depends on.
    """
    employee_ids = fetch_real_employee_ids(session)

    effective_mix = dict(mix)
    if "diagnose" in effective_mix:
        ticket_ids = fetch_real_ticket_ids(session)
        if not ticket_ids:
            effective_mix["create_ticket"] = effective_mix.get("create_ticket", 0.0) + effective_mix.pop("diagnose")
    else:
        ticket_ids = []

    queue_ids = fetch_real_queue_ids(session) if "create_ticket" in effective_mix else []
    chat_questions = load_real_chat_questions() if "chat" in effective_mix else []
    ticket_descriptions = load_real_ticket_descriptions() if "create_ticket" in effective_mix else []
    # "diagnose" needs its OWN employee pool, not `employee_ids` -- see
    # `fetch_real_diagnose_capable_employee_ids`'s docstring for why most
    # of the general active-employee pool would 403 against the real
    # ticket.diagnose staff-only authz check.
    diagnose_employee_ids = (
        fetch_real_diagnose_capable_employee_ids(session) if "diagnose" in effective_mix else []
    )

    kinds = list(effective_mix.keys())
    weights = list(effective_mix.values())

    items: list[WorkloadItem] = []
    for _ in range(size):
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        if kind == "chat":
            items.append(
                WorkloadItem(
                    kind="chat",
                    employee_id=rng.choice(employee_ids),
                    payload={"message": rng.choice(chat_questions)},
                )
            )
        elif kind == "create_ticket":
            text = rng.choice(ticket_descriptions)
            items.append(
                WorkloadItem(
                    kind="create_ticket",
                    employee_id=rng.choice(employee_ids),
                    payload={
                        "subject": text[:200],
                        "body": text,
                        "type": "incident" if rng.random() < 0.7 else "service_request",
                        "priority": rng.choice(["low", "medium", "medium", "high", "urgent"]),
                        "queue_id": rng.choice(queue_ids),
                    },
                )
            )
        else:  # "diagnose"
            items.append(
                WorkloadItem(
                    kind="diagnose",
                    employee_id=rng.choice(diagnose_employee_ids),
                    path_params={"ticket_id": rng.choice(ticket_ids)},
                )
            )
    return items


# ---------------------------------------------------------------------------
# Bounded-concurrency async HTTP client scaffold
# ---------------------------------------------------------------------------

# Real endpoint shapes, from routers/chat.py and routers/tickets.py --
# kept here as the one place this harness's workload "kind" names map to
# real HTTP method+path, so Task 2 (or a future mix change) only has to
# update one table.
_METHOD_PATH: dict[str, tuple[str, str]] = {
    "chat": ("POST", "/chat"),
    "create_ticket": ("POST", "/tickets"),
    "diagnose": ("POST", "/tickets/{ticket_id}/diagnose"),
}


@dataclass(frozen=True)
class RequestResult:
    """Outcome of firing one `WorkloadItem`. Timestamps are seconds
    since the run started (`time.perf_counter()`-based, client-side --
    per the plan doc, Task 2 cross-checks these against real `AgentRun`/
    `Span` server-side timestamps, never trusting the client timer alone).
    `error` is set (and `status_code` is `None`) if the request itself
    raised (connection refused, timeout, etc.) -- recorded as data, never
    raised further, so one bad request can't abort an entire run.
    """

    item: WorkloadItem
    scheduled_at: float
    started_at: float
    finished_at: float
    status_code: int | None
    error: str | None

    @property
    def latency_seconds(self) -> float:
        return self.finished_at - self.started_at


class BoundedConcurrencyLoadClient:
    """Fires a list of `WorkloadItem`s at `base_url` according to a list
    of `Arrival`s (one arrival per item, zipped by position), never
    allowing more than `max_concurrency` requests in flight at once.

    This is the scaffold Task 2 will point at the real running Docker
    Compose API (`base_url` = that stack's real address) -- Task 1 does
    not run an actual load test against a live stack (out of this task's
    scope, per the dispatching instructions). Its concurrency-bounding
    and request-firing logic is instead proven correct here against a
    real local ASGI app via `transport=httpx.ASGITransport(app=...)`
    (see `test_load_test.py`) -- a genuine in-process HTTP round trip
    through the real ASGI protocol, not a mocked network call -- so this
    class is already verified working before Task 2 points it at Docker.

    Implementation: one `asyncio.Semaphore(max_concurrency)` shared by
    every scheduled request coroutine, acquired only *after* each
    coroutine has slept until its own arrival time -- so the semaphore
    bounds actual concurrent in-flight requests, not merely how many
    requests get scheduled to wait for their turn.
    """

    def __init__(
        self,
        base_url: str,
        *,
        max_concurrency: int = 20,
        timeout_seconds: float = 30.0,
        employee_header: str = "X-Debug-Employee-Id",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._base_url = base_url
        self._max_concurrency = max_concurrency
        self._timeout_seconds = timeout_seconds
        self._employee_header = employee_header
        # `transport=None` lets httpx.AsyncClient make a real outbound
        # TCP connection (Task 2's live-stack use); tests pass a real
        # httpx.ASGITransport instead, per the class docstring.
        self._transport = transport

    async def run_schedule(self, arrivals: list[Arrival], items: list[WorkloadItem]) -> list[RequestResult]:
        if len(arrivals) != len(items):
            raise ValueError(
                f"arrivals ({len(arrivals)}) and items ({len(items)}) must be the same length -- "
                "one arrival per workload item, zipped by schedule position"
            )

        semaphore = asyncio.Semaphore(self._max_concurrency)
        results: list[RequestResult | None] = [None] * len(items)
        run_start = time.perf_counter()

        async def fire(index: int, arrival: Arrival, item: WorkloadItem) -> None:
            delay = arrival.t - (time.perf_counter() - run_start)
            if delay > 0:
                await asyncio.sleep(delay)
            async with semaphore:
                results[index] = await self._send_one(client, arrival, item, run_start)

        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout_seconds, transport=self._transport
        ) as client:
            await asyncio.gather(*(fire(i, a, it) for i, (a, it) in enumerate(zip(arrivals, items))))

        return [r for r in results if r is not None]

    async def _send_one(
        self, client: httpx.AsyncClient, arrival: Arrival, item: WorkloadItem, run_start: float
    ) -> RequestResult:
        method, path_template = _METHOD_PATH[item.kind]
        path = path_template.format(**item.path_params)
        headers = {self._employee_header: str(item.employee_id)}
        started_at = time.perf_counter() - run_start
        try:
            response = await client.request(method, path, json=item.payload, headers=headers)
            finished_at = time.perf_counter() - run_start
            return RequestResult(
                item=item,
                scheduled_at=arrival.t,
                started_at=started_at,
                finished_at=finished_at,
                status_code=response.status_code,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 -- a load client must record a
            # failed request as data (connection refused, timeout, etc.),
            # never let one bad request crash the whole scheduled run --
            # mirrors eval_worker.py's own "record the error, keep going"
            # convention for batch harnesses.
            finished_at = time.perf_counter() - run_start
            return RequestResult(
                item=item,
                scheduled_at=arrival.t,
                started_at=started_at,
                finished_at=finished_at,
                status_code=None,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# LoadTestRun manifest persistence
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    """The real current commit sha, via `git rev-parse HEAD` -- mirrors
    `eval_worker.py`'s own `_git_commit()` helper byte-for-byte (not
    imported from there: that function is module-private to a different
    concern, and every other cross-cutting helper this small in this
    codebase is duplicated locally rather than exported, e.g.
    `_fetch_chunk_texts`/`_fetch_superseded_chunk_ids` in
    `eval_retrieval.py`). Never raises -- returns "unknown" if this isn't
    a git checkout or git isn't on PATH, so a broken git toolchain can
    never take down a load-test run over a purely cosmetic field.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 -- see docstring: must never raise
        return "unknown"


def build_run_config_json(
    seed: int,
    config: NHPPConfig,
    mix: dict[str, float] = DEFAULT_WORKLOAD_MIX,
) -> str:
    """Serializes the exact configuration needed to reproduce a run into
    the JSON string stored in `LoadTestRun.config_json`, per the phase
    plan's schema note ("seed, duration, target rps, burst schedule").
    `target_rps` here is `mean_intensity_rps(config)` -- the REAL
    empirical time-average the configured base/peak/shape produces, not
    a separately-specified (and potentially inconsistent) input value.
    """
    return json.dumps(
        {
            "seed": seed,
            "duration_seconds": config.duration_seconds,
            "base_rate_rps": config.base_rate_rps,
            "peak_rate_rps": config.peak_rate_rps,
            "target_rps": mean_intensity_rps(config),
            "peak_hours": list(config.peak_hours),
            "peak_width_hours": config.peak_width_hours,
            "burst_schedule": list(config.burst_schedule),
            "burst_mean_size": config.burst_mean_size,
            "burst_jitter_seconds": config.burst_jitter_seconds,
            "workload_mix": mix,
        },
        sort_keys=True,
    )


def create_load_test_run(session: Session, *, config_json: str, git_commit: str | None = None) -> LoadTestRun:
    """Inserts and flushes a new `LoadTestRun` row marking the start of a
    run (`completed_at` stays NULL until `complete_load_test_run` is
    called). `git_commit` defaults to the real current commit via
    `_git_commit()` -- a caller only needs to pass it explicitly in a
    test that wants a fixed, known value.
    """
    run = LoadTestRun(config_json=config_json, git_commit=git_commit or _git_commit())
    session.add(run)
    session.flush()
    return run


def complete_load_test_run(
    session: Session,
    run: LoadTestRun,
    *,
    result_summary_json: str,
    raw_results_path: str,
) -> None:
    """Marks `run` complete: sets `completed_at` to the real current UTC
    time plus the run's result summary and raw-results file pointer.
    Task 2 calls this once a real run against the live stack finishes
    and its raw per-request results have actually been written to disk
    under `eval/results/...` -- `raw_results_path` should name a file
    that exists by the time this is called, though this function itself
    does not verify that (it only records metadata; Task 2's own
    reproducible-command contract is what guarantees the file is real).
    """
    run.completed_at = datetime.now(timezone.utc)
    run.result_summary_json = result_summary_json
    run.raw_results_path = raw_results_path
    session.flush()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Standalone dry-run CLI: `uv run --package resolvegrid-api python -m
    resolvegrid_api.load_test --seed 42 --duration-seconds 3600 --base-rps
    0.01 --peak-rps 0.05`.

    Deliberately does NOT fire any HTTP requests against a live stack --
    that is Task 2's job, against the real running Docker Compose API.
    This prints the sampled arrival count/shape and a workload-mix
    preview against whatever is really seeded in the configured dev DB,
    useful for sanity-checking a chosen seed/duration/rps/burst-schedule
    combination before Task 2 spends real wall-clock time and real model
    calls running it for real.
    """
    import argparse

    from resolvegrid_api.db import DATABASE_URL
    from sqlalchemy import create_engine

    parser = argparse.ArgumentParser(description="ResolveGrid load-test harness (Task 1 dry run, no live HTTP calls)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-seconds", type=float, default=3600.0)
    parser.add_argument("--base-rps", type=float, default=0.01)
    parser.add_argument("--peak-rps", type=float, default=0.05)
    parser.add_argument("--burst-schedule", type=float, nargs="*", default=[])
    parser.add_argument("--workload-size", type=int, default=100)
    args = parser.parse_args()

    config = NHPPConfig(
        duration_seconds=args.duration_seconds,
        base_rate_rps=args.base_rps,
        peak_rate_rps=args.peak_rps,
        burst_schedule=tuple(args.burst_schedule),
    )
    arrivals = sample_nhpp_arrivals(config, seed=args.seed)
    burst_count = sum(1 for a in arrivals if a.is_burst)
    print(
        f"Sampled {len(arrivals)} arrivals over {config.duration_seconds:.0f}s "
        f"({burst_count} from burst injection), empirical mean rate "
        f"{mean_intensity_rps(config):.5f} req/s."
    )

    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        try:
            rng = random.Random(args.seed)
            items = build_workload_mix(session, rng, args.workload_size)
            counts: dict[str, int] = {}
            for item in items:
                counts[item.kind] = counts.get(item.kind, 0) + 1
            print(f"Workload mix preview ({args.workload_size} items): {counts}")
        except ValueError as exc:
            print(f"Workload mix preview skipped -- {exc}")
    engine.dispose()


if __name__ == "__main__":
    main()

"""Authorization-aware metadata filtering for knowledge retrieval (Phase 7
Task 5).

Turns a resolved `Principal` (see `resolvegrid_api.deps.get_principal`) into
a concrete, SQL-bakeable predicate over `Document.access_scope_tags` --
`build_authz_filter()` -- so `resolvegrid_api.retrieval.vector_search`/
`lexical_search` can require it as a mandatory parameter and fold it
straight into their `WHERE` clause (JOIN `Chunk` -> `DocumentVersion` ->
`Document`, per the prior review's note that `access_scope_tags` lives on
`Document`, not `Chunk`). This is a filter *representation*, not a
SQLAlchemy `Session`-bound query -- callers still execute it themselves.

Principal -> allowed-tags mapping (and why)
--------------------------------------------
`packages/authz`'s `Principal` (resolvegrid_authz.Principal) carries only
`employee_id` and a tuple of `RoleGrant`s -- there is no "entitlements"
-style field on `Principal`, nor on the `Employee` model it's ultimately
backed by (`resolvegrid_api.models.org.Employee`). Adding a new
entitlements schema column is out of scope for this task, so this module
reuses what already exists (`Employee.department_id`, `RoleAssignment`
-derived roles) through the *same* `authorize()` entry point
`routers/tickets.py` already calls
(`packages/authz/src/resolvegrid_authz/policy.py`) -- not a parallel,
ad-hoc authorization mechanism. A new action, `"knowledge.retrieve"`, was
added to that policy's `_SELF_SCOPED_ACTIONS` for this purpose: it reuses
the exact admin / department-scoped / self-scoped `Decision` shape
`ticket.list` already produces, rather than inventing a bespoke retrieval
policy.

Concretely, `authorize(principal, "knowledge.retrieve")` is called, then:

- Global admin (`Decision.department_ids is None and Decision.employee_id
  is None`) -> `AuthzFilter(unrestricted=True, ...)`: every chunk is
  visible regardless of its Document's `access_scope_tags`.
- A department-scoped grant (analyst/approver with `scope="department"`)
  -> `Decision.department_ids` is a tuple of `Department.id` values; each
  is resolved to its `Department.name` and normalized into a tag
  (lowercased, spaces -> underscores -- e.g. "Platform Engineering" ->
  "platform_engineering", matching the seeded department names in
  `apps/api/src/resolvegrid_api/seed.py`'s `_DEPARTMENTS`). The allowed
  -tags set is exactly the set of departments the grant covers.
- No matching role grant (`Decision.employee_id` is set, i.e. self-scope)
  -> falls back to the requesting employee's own home department
  (`Employee.department_id`), normalized the same way. Rationale: a
  Document's `access_scope_tags` describe *which department's knowledge*
  a chunk belongs to, not an individually-owned resource the way a Ticket
  is -- so "self-scope" for knowledge retrieval is interpreted as
  "documents scoped to my own department", the closest existing concept
  on the Employee model. An employee with no department
  (`department_id is None`) or who doesn't exist gets an *empty*
  allowed-tags set (fail closed: they see only unscoped/public documents,
  never everything).
- A Document with an **empty** `access_scope_tags` array is treated as
  unscoped/public (visible to everyone, regardless of `AuthzFilter`) --
  the convention this module establishes for public-attributable vendor
  docs (`Document.source_type == "public"`), which have no single
  department to scope them to. This policy choice is enforced in the SQL
  itself (see `retrieval.py`), not here, but is documented here since
  nothing upstream pins it down.
"""

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api.models.org import Department, Employee
from resolvegrid_authz import Principal, authorize


@dataclass(frozen=True)
class AuthzFilter:
    """Concrete allowed-access-scope-tags predicate for knowledge retrieval.

    `unrestricted=True` means "skip the tag filter entirely" (admin) --
    `allowed_tags` is meaningless in that case and left empty. Otherwise, a
    chunk is visible iff its Document's `access_scope_tags` is empty
    (public) or overlaps `allowed_tags`.
    """

    unrestricted: bool
    allowed_tags: frozenset[str] = field(default_factory=frozenset)


def normalize_department_tag(department_name: str) -> str:
    """Normalize a Department name into an access-scope tag, e.g.
    "Platform Engineering" -> "platform_engineering".

    The single normalization rule shared by every place a Department name
    becomes an access-scope tag (this module now; the seed/ingestion
    corpus later), so a department's tag is always derivable the same way.
    """
    return department_name.strip().lower().replace(" ", "_")


def build_authz_filter(principal: Principal, session: Session) -> AuthzFilter:
    """Map `principal` to a concrete `AuthzFilter`, per this module's
    docstring. Always returns a filter (never raises for an unknown/absent
    employee) -- an employee record that can't be resolved just yields an
    empty allowed-tags set (fail closed), not an error, so a caller doesn't
    need special-case error handling here on top of `authorize()`'s own
    allow/deny.
    """
    decision = authorize(principal, "knowledge.retrieve")

    if not decision.allowed:
        # authorize() only denies "knowledge.retrieve" for a truly unknown
        # action (see policy.py) -- since it's registered as self-scoped,
        # this branch is unreachable in practice, but fail closed rather
        # than silently granting access if that ever changes.
        return AuthzFilter(unrestricted=False, allowed_tags=frozenset())

    if decision.department_ids is None and decision.employee_id is None:
        return AuthzFilter(unrestricted=True)

    if decision.department_ids is not None:
        department_ids = decision.department_ids
    else:
        employee = session.get(Employee, decision.employee_id)
        department_ids = (employee.department_id,) if employee and employee.department_id is not None else ()

    if not department_ids:
        return AuthzFilter(unrestricted=False, allowed_tags=frozenset())

    department_names = session.scalars(
        select(Department.name).where(Department.id.in_(department_ids))
    ).all()
    allowed_tags = frozenset(normalize_department_tag(name) for name in department_names)
    return AuthzFilter(unrestricted=False, allowed_tags=allowed_tags)

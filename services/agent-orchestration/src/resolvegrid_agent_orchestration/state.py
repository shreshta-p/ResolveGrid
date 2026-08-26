"""Graph state shape for the Phase 6 agent workflow.

Deliberately minimal: no `data_scope_tags`/retrieval-related fields yet.
Those belong to Phase 7, once a real knowledge base exists to retrieve
against -- adding them now would be dead, untested surface area.
"""

from typing import TypedDict


class AgentState(TypedDict):
    thread_id: str
    principal_employee_id: int | None
    input_text: str
    intent: str | None
    risk_level: str | None
    output_text: str | None
    error: str | None

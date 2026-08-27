from resolvegrid_authz import Principal, RoleGrant, authorize


def test_admin_has_unrestricted_access():
    principal = Principal(employee_id=1, roles=(RoleGrant(role="admin", scope="global"),))
    decision = authorize(principal, "directory.list_employees")
    assert decision.allowed is True
    assert decision.department_ids is None
    assert decision.employee_id is None


def test_analyst_is_restricted_to_own_department():
    principal = Principal(
        employee_id=2, roles=(RoleGrant(role="analyst", scope="department", scope_id=5),)
    )
    decision = authorize(principal, "directory.list_employees")
    assert decision.allowed is True
    assert decision.department_ids == (5,)


def test_approver_is_restricted_to_own_department():
    principal = Principal(
        employee_id=3, roles=(RoleGrant(role="approver", scope="department", scope_id=9),)
    )
    decision = authorize(principal, "directory.list_employees")
    assert decision.allowed is True
    assert decision.department_ids == (9,)


def test_employee_with_no_role_grant_sees_only_self():
    principal = Principal(employee_id=7, roles=())
    decision = authorize(principal, "directory.list_employees")
    assert decision.allowed is True
    assert decision.employee_id == 7


def test_unknown_action_is_denied():
    principal = Principal(employee_id=1, roles=(RoleGrant(role="admin", scope="global"),))
    decision = authorize(principal, "directory.delete_everything")
    assert decision.allowed is False


def test_admin_grant_dominates_regardless_of_order():
    principal = Principal(
        employee_id=4,
        roles=(
            RoleGrant(role="analyst", scope="department", scope_id=5),
            RoleGrant(role="admin", scope="global"),
        ),
    )
    decision = authorize(principal, "directory.list_employees")
    assert decision.allowed is True
    assert decision.department_ids is None
    assert decision.employee_id is None


def test_multiple_department_grants_are_all_included():
    principal = Principal(
        employee_id=6,
        roles=(
            RoleGrant(role="analyst", scope="department", scope_id=5),
            RoleGrant(role="approver", scope="department", scope_id=9),
        ),
    )
    decision = authorize(principal, "directory.list_employees")
    assert decision.allowed is True
    assert decision.department_ids == (5, 9)


def test_department_grant_with_missing_scope_id_is_ignored_not_unrestricted():
    principal = Principal(
        employee_id=8, roles=(RoleGrant(role="analyst", scope="department", scope_id=None),)
    )
    decision = authorize(principal, "directory.list_employees")
    assert decision.allowed is True
    assert decision.department_ids is None
    assert decision.employee_id == 8


def test_admin_grant_with_wrong_scope_is_ignored():
    principal = Principal(
        employee_id=9, roles=(RoleGrant(role="admin", scope="department", scope_id=3),)
    )
    decision = authorize(principal, "directory.list_employees")
    assert decision.allowed is True
    assert decision.employee_id == 9


def test_ticket_list_is_self_scoped_for_employee_with_no_grant():
    principal = Principal(employee_id=10, roles=())
    decision = authorize(principal, "ticket.list")
    assert decision.allowed is True
    assert decision.employee_id == 10


def test_ticket_view_is_department_scoped_for_analyst():
    principal = Principal(employee_id=11, roles=(RoleGrant(role="analyst", scope="department", scope_id=3),))
    decision = authorize(principal, "ticket.view")
    assert decision.allowed is True
    assert decision.department_ids == (3,)


def test_ticket_transition_denied_for_plain_employee():
    principal = Principal(employee_id=12, roles=())
    decision = authorize(principal, "ticket.transition")
    assert decision.allowed is False


def test_ticket_transition_allowed_for_department_scoped_analyst():
    principal = Principal(employee_id=13, roles=(RoleGrant(role="analyst", scope="department", scope_id=4),))
    decision = authorize(principal, "ticket.transition")
    assert decision.allowed is True
    assert decision.department_ids == (4,)


def test_ticket_transition_allowed_for_admin():
    principal = Principal(employee_id=14, roles=(RoleGrant(role="admin", scope="global"),))
    decision = authorize(principal, "ticket.transition")
    assert decision.allowed is True


def test_knowledge_retrieve_is_self_scoped_for_employee_with_no_grant():
    # Phase 7 Task 5: knowledge.retrieve reuses the same self-scoped
    # Decision shape as ticket.list/directory.list_employees -- the
    # retrieval-specific interpretation of a self-scoped Decision lives in
    # apps/api/src/resolvegrid_api/retrieval_authz.py, not here.
    principal = Principal(employee_id=15, roles=())
    decision = authorize(principal, "knowledge.retrieve")
    assert decision.allowed is True
    assert decision.employee_id == 15
    assert decision.department_ids is None


def test_knowledge_retrieve_is_department_scoped_for_analyst():
    principal = Principal(
        employee_id=16, roles=(RoleGrant(role="analyst", scope="department", scope_id=7),)
    )
    decision = authorize(principal, "knowledge.retrieve")
    assert decision.allowed is True
    assert decision.department_ids == (7,)


def test_knowledge_retrieve_is_unrestricted_for_admin():
    principal = Principal(employee_id=17, roles=(RoleGrant(role="admin", scope="global"),))
    decision = authorize(principal, "knowledge.retrieve")
    assert decision.allowed is True
    assert decision.department_ids is None
    assert decision.employee_id is None

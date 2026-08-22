from resolvegrid_authz import Principal, RoleGrant, authorize


def test_admin_has_unrestricted_access():
    principal = Principal(employee_id=1, roles=[RoleGrant(role="admin", scope="global")])
    decision = authorize(principal, "directory.list_employees")
    assert decision.allowed is True
    assert decision.filter == {}


def test_analyst_is_restricted_to_own_department():
    principal = Principal(
        employee_id=2, roles=[RoleGrant(role="analyst", scope="department", scope_id=5)]
    )
    decision = authorize(principal, "directory.list_employees")
    assert decision.allowed is True
    assert decision.filter == {"department_id": 5}


def test_approver_is_restricted_to_own_department():
    principal = Principal(
        employee_id=3, roles=[RoleGrant(role="approver", scope="department", scope_id=9)]
    )
    decision = authorize(principal, "directory.list_employees")
    assert decision.allowed is True
    assert decision.filter == {"department_id": 9}


def test_employee_with_no_role_grant_sees_only_self():
    principal = Principal(employee_id=7, roles=[])
    decision = authorize(principal, "directory.list_employees")
    assert decision.allowed is True
    assert decision.filter == {"employee_id": 7}


def test_unknown_action_is_denied():
    principal = Principal(employee_id=1, roles=[RoleGrant(role="admin", scope="global")])
    decision = authorize(principal, "directory.delete_everything")
    assert decision.allowed is False

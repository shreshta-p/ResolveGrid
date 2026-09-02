"""Operational adapters: DB-backed implementations of tool-contract actions
that need real writes against `apps/api`'s own ORM models (Phase 9 Task 3).

See `entitlements.py`'s module docstring for why this lives here as a
plain module inside `apps/api` rather than as a new
`services/operational-adapters` workspace package -- that was the original
plan-mode sketch, amended before implementation.
"""

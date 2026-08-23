"""Tamper-evident audit logging.

Each AuditLog row's record_hash is derived from the previous row's hash plus
this row's own fields, forming a hash chain: altering any historical row's
content without recomputing every subsequent hash makes the tampering
detectable via verify_chain_integrity().

Known limitation (documented, not fixed in Phase 3): record_audit_event reads
the "previous" row and writes the new row within the caller's own session/
transaction with no explicit row lock. Under genuine concurrent writers this
has a race window (two transactions could both read the same "last" row and
compute conflicting chains). Acceptable for Phase 3's single-writer-at-a-time
walking-skeleton scope; revisit with SELECT ... FOR UPDATE or a serializable
isolation level if concurrent ticket-mutation load becomes real.
"""

import hashlib
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api.models import AuditLog


def _compute_record_hash(
    previous_hash: str | None,
    actor_type: str,
    actor_id: int | None,
    action: str,
    entity_type: str,
    entity_id: int,
    before_json: str | None,
    after_json: str | None,
    metadata_json: str | None,
) -> str:
    payload = json.dumps(
        {
            "previous_hash": previous_hash,
            "actor_type": actor_type,
            "actor_id": actor_id,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "before_json": before_json,
            "after_json": after_json,
            "metadata_json": metadata_json,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_audit_event(
    session: Session,
    *,
    actor_type: str,
    actor_id: int | None,
    action: str,
    entity_type: str,
    entity_id: int,
    before: dict | None = None,
    after: dict | None = None,
    metadata: dict | None = None,
) -> AuditLog:
    previous = session.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(1)).first()
    previous_hash = previous.record_hash if previous is not None else None

    before_json = json.dumps(before, sort_keys=True) if before is not None else None
    after_json = json.dumps(after, sort_keys=True) if after is not None else None
    metadata_json = json.dumps(metadata, sort_keys=True) if metadata is not None else None

    record_hash = _compute_record_hash(
        previous_hash, actor_type, actor_id, action, entity_type, entity_id, before_json, after_json, metadata_json
    )

    entry = AuditLog(
        actor_type=actor_type,
        actor_id=actor_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        before_json=before_json,
        after_json=after_json,
        metadata_json=metadata_json,
        record_hash=record_hash,
        previous_record_hash=previous_hash,
    )
    session.add(entry)
    session.flush()
    return entry


def verify_chain_integrity(session: Session) -> bool:
    """Recompute every row's hash from its stored fields and confirm the
    chain is unbroken.

    Loads the entire audit_log table into memory -- fine for Phase 3's
    walking-skeleton scope, but will need incremental/checkpointed
    verification (verify only rows after a last-known-good point) once row
    counts grow large in a real deployment.
    """
    rows = session.scalars(select(AuditLog).order_by(AuditLog.id.asc())).all()
    previous_hash: str | None = None
    for row in rows:
        expected_hash = _compute_record_hash(
            previous_hash, row.actor_type, row.actor_id, row.action,
            row.entity_type, row.entity_id, row.before_json, row.after_json,
            row.metadata_json,
        )
        if row.previous_record_hash != previous_hash:
            return False
        if row.record_hash != expected_hash:
            return False
        previous_hash = row.record_hash
    return True

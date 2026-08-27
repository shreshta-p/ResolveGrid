# Kestrel Platform Engineering On-Call Escalation Policy

This policy defines how production incidents are escalated within the
Platform Engineering department.

## On-Call Rotation

Platform Engineering maintains a weekly on-call rotation. The primary
on-call engineer is the first responder for every paging alert; a
secondary on-call engineer is paged automatically if the primary does not
acknowledge within 10 minutes.

## Severity Levels and Response Times

Sev-1 incidents (full service outage) require acknowledgment within 5
minutes and an initial status update within 15 minutes. Sev-2 incidents
(significant degradation, partial outage) require acknowledgment within
15 minutes. Sev-3 incidents (minor, non-customer-facing issues) are
handled during normal business hours and do not page the on-call
rotation.

## Escalation Path

If a Sev-1 incident is not mitigated within 30 minutes, the on-call
engineer escalates to the Platform Engineering duty manager, who can pull
in additional engineers or declare a formal incident bridge. Incidents
touching authentication or data integrity are escalated to Security in
parallel, regardless of severity level.

## Postmortems

Every Sev-1 and Sev-2 incident requires a written postmortem within 5
business days, covering root cause, timeline, and follow-up action items.
Postmortems are blameless and are reviewed in the department's weekly
engineering sync.

# Phase 14 Task 5 — Manual-vs-AI benchmark (claim 2): outcome

**Status: ATTEMPTED, then deliberately abandoned by the human operator. Claim 2
is reported as genuinely unmeasured, not met/not-met.**

## What was planned

Per `protocol.md` in this same directory: 5 real tickets (materialized from
this project's own existing golden diagnostic-eval fixtures, IDs 1457-1461)
were set up for the human operator to resolve manually (no AI assistance),
timing themselves, to produce a real baseline for comparison against the
AI-assisted `/tickets/{id}/diagnose` wall-clock time on the same tickets.

## What actually happened

The operator began the session on ticket 1457, then reached ticket 1458
(the "coffee machine" case) and, after a wrong debug-employee-id attempt,
clicked the "Summarize" button on the ticket page — an AI-assisted action,
which is disallowed under the protocol's own start/stop rules (no `/chat`,
no "Diagnose," no "Summarize" during the timed manual window). This
contaminated ticket 1458's manual attempt. No elapsed times or composed
answers were ever recorded for any of the 5 tickets before this point.

After this, the operator stated the real, underlying reason they did not
want to continue: **these tickets are synthetic/fabricated (drawn from this
project's own golden-eval fixtures), and a human cannot meaningfully
"resolve" a made-up ticket the way they would a real one** — there is no
genuine stake, no real requester to satisfy, and no real-world context that
would let a person engage with the task the way an actual IT analyst
engages with an actual incident. The operator chose to stop rather than
produce timings that would not represent real task-completion behavior.

## Why this is reported as a real finding, not just "declined to participate"

This project's standing rule (`plan.md`'s Outcomes section) is to report
actual outcomes honestly, including when a target cannot be assessed. The
original protocol (`protocol.md` §0, written before this session) already
flagged that no human subject was available in this environment to measure
claim 2, and the human operator agreed to attempt a real session anyway.
That attempt surfaced a MORE SPECIFIC and arguably more important
limitation than "no human available": **even with a willing human, a
synthetic/fabricated ticket corpus does not create the genuine task
conditions a manual-vs-AI timing comparison needs to be meaningful.** A
real benchmark of this kind would need real historical tickets with real
stakes, not fixtures authored for an evaluation harness. This is a real,
substantive constraint of this project's synthetic-data-only environment
(`docs/SECURITY.md`'s "not real data" disclosure applies here too), not a
reflection on the operator.

## Claim 2 verdict

**Claim 2 (~80% reduction in manual information-gathering time): UNMEASURED.**
No valid manual-timing data exists for any of the 5 tickets. This is
reported explicitly as unmeasured rather than estimated, assumed, or
silently omitted, per this phase's standing honesty rule. Unlike claims
1/3/4/5 (all measured with real numbers, met or not met), claim 2 could not
be assessed in this environment at all -- with or without a human
operator -- because no real (non-synthetic) ticket corpus with genuine
task stakes exists here to measure against.

## Disposition of the prepared materials

- `protocol.md` and `kb_export_for_manual_lookup.txt` are kept in this
  directory as a record of the real, careful protocol design (sampling,
  start/stop rules, correctness thresholds, disclosed bias/limitations) --
  useful if a future phase gains access to a real (non-synthetic) ticket
  corpus and wants to re-attempt this benchmark properly.
- The 5 throwaway fixture tickets (1457-1461), their messages, the 2
  throwaway KB documents (552-553, with their chunks/embeddings), and 5
  throwaway queues were deleted from the dev DB after this session ended,
  since they were materialized only for this attempt and are no longer
  needed. One throwaway fixture employee (id 2572) could not be deleted --
  it is referenced by a real, hash-chained `AuditLog` row (`actor_id`,
  the `ticket.reopen` action recorded for ticket 1460) and this project's
  audit log is append-only by design (the same protection that already
  applies to ~22 other real employees in this dev DB from prior real
  activity). It remains as harmless residue, consistent with established
  project precedent, not a bug.

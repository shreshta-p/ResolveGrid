# Phase 14 Task 5 -- Manual-vs-AI benchmark protocol (claim 2)

**Status: PREPARATION ONLY.** This document designs the protocol and
prepares the environment. It does NOT contain measured times yet. The
manual-lookup run requires the project's real human operator
(shreshta88.p@gmail.com) to actually perform it; the AI-assisted comparison
run happens afterward, as a separate step, once the human's real times are
recorded. Neither has been run as of this writing (2026-11-11).

Evaluates plan.md's claim 2: "~80% reduction in manual information-
gathering time." Per `PLAN_APPROVED.md` §12 and the Phase 14 plan's Task 5,
this requires: defined sampling, start/stop rules, correctness threshold,
sample size, failure treatment, and documented bias/limitations.

## 0. Real, upfront environmental findings (read before running anything)

These were discovered while preparing this protocol and materially shape
its design. Stated plainly rather than hidden, per this phase's "nothing
mocked, report reality" standing directive:

1. **The live dev DB had no realistic ticket corpus.** Querying
   `SELECT * FROM ticket` returned exactly 5 rows, all subject="Spam",
   body="x", in a queue literally named "Ticket Test Queue" (IDs
   1438-1442) -- smoke-test residue from an earlier automated run, not a
   seeded demo corpus. `apps/api/src/resolvegrid_api/seed.py` only seeds
   org/identity data (employees/departments/teams), never tickets. There
   is no dedicated "ticket seed generator" in this codebase.

2. **The real, realistic, human-reviewed ticket scenarios live in
   `eval/golden/diagnostic_v1.jsonl`'s fixture builders**
   (`apps/api/src/resolvegrid_api/diagnostic_eval.py`), which Task 4's own
   eval run already uses as this phase's golden diagnostic set. Each case
   has a real, deterministic, self-contained Ticket+TicketMessage fixture
   (and for some cases a real seeded KB Document+Chunk+Embedding) with a
   human-reviewed rubric defining what a correct answer looks like. These
   fixtures are normally built, used for one eval run, then deleted
   (`cleanup()`) -- they do not normally persist as browsable DB rows.

3. **Resolution: 5 of these 6 golden fixtures were materialized as real,
   persisted `Ticket`/`TicketMessage` (and where applicable `Document`/
   `Chunk`/`Embedding`) rows** in the dev DB, using the project's own real,
   unmodified builder functions (`_build_vpn_access_issue`,
   `_build_no_kb_evidence_honest_abstention`,
   `_build_entitlement_request_no_false_claim`,
   `_build_audit_history_grounding`,
   `_build_email_password_reset_categorization`), just withholding each
   builder's own `cleanup()` call so the rows stick around. This is the
   same real code path Task 4 already relies on as the project's golden
   diagnostic sample -- not fabricated data, and not a new, untested
   generator. The 6th case (`similar_past_ticket_grounding`) was dropped
   because its topic (VPN, again) duplicates `vpn_access_issue`'s topic --
   dropping it maximizes topic diversity across the 5 selected, per this
   task's explicit "mix of topics/complexity" requirement. See the
   one-off script this used, archived at the bottom of this doc (§8) for
   exact reproducibility.

4. **The ticket detail page UI does not render the ticket's description
   at all.** `apps/web/app/tickets/[id]/page.tsx` only displays
   subject/status/priority (and buttons). The description/body lives only
   in `TicketMessage.body`, and there is no `GET /tickets/{id}/messages`
   endpoint or any UI surface for it. **Good news for blind timing**: the
   AI "Diagnose" and "Summarize" panels are strictly click-to-run
   (`<button onClick={diagnose}>` / `<button onClick={summarize}>`, no
   `useEffect` auto-fires either) -- confirmed by reading the component.
   So nothing AI-generated is ever pre-rendered; the real gap is the
   *opposite* problem, the description itself isn't shown anywhere live.
   **Workaround**: since no endpoint exposes it, and writing a new one is
   out of scope for a preparation-only task, each ticket's real subject +
   description is recorded directly in §3 below for the human to read as
   the "ticket just came in" moment that starts the clock. This is
   necessary, not a convenience shortcut -- documented as a limitation in
   §6.

5. **There is no non-AI knowledge-base search/browse surface in the
   product at all.** No `/documents`, `/kb`, or `/search` route exists in
   `apps/api/src/resolvegrid_api/routers/*.py`; `apps/web` has no KB page.
   The only paths to KB content are AI-assisted (`POST /chat`, `POST
   /tickets/{id}/diagnose`), both disallowed in the manual condition. This
   is reported here explicitly as its own finding, per this task's
   instruction to do so if found. **Workaround**: a flat, verbatim,
   read-only text export of every real KB document/chunk currently in the
   DB was produced via `docker exec resolvegrid-postgres ... psql`
   (`kb_export_for_manual_lookup.txt`, same directory as this file) and is
   the sanctioned manual-lookup tool, standing in for "the company wiki."
   This is a real limitation (a true new analyst would use an actual
   search tool, not Ctrl+F one flat file) -- see §6.

6. **There is also no non-AI way to check an employee's current
   entitlements or a ticket's own audit/reopen history.**
   `apps/api/src/resolvegrid_api/routers/directory.py` has no
   `/employees/{id}/entitlements` route, and no router anywhere exposes
   `AuditLog`/`TicketStateTransition` rows. Only the AI diagnostic
   specialist's internal tools (`get_employee_entitlements`,
   `get_ticket_state_and_history`) reach this data. Two of the five
   selected tickets (#1459, #1460) specifically require this kind of
   fact. **Workaround**: for exactly these two tickets, the sanctioned
   manual-lookup method includes one real, read-only `docker exec
   resolvegrid-postgres ... psql` query against `employee_entitlement` /
   `audit_log` (the same safe, non-secret-printing invocation style used
   throughout this prep), given as part of the human's instructions in
   §4. This is an unusual ask of a "manual" analyst (a real one would use
   an admin UI, not SQL) -- flagged plainly as a limitation in §6, not
   smoothed over.

## 1. Sample (5 tickets -- the full population, not a cherry-picked subset)

The entire realistic corpus prepared for this benchmark is these 5 tickets
-- not a subset selected from a larger pool, so there is no cherry-picking
question: every available real, human-reviewed, KB-matched scenario is
used. They are the project's own existing golden diagnostic eval cases
(Task 4's own sample), chosen for maximum topic/complexity diversity (5 of
the 6 available cases, dropping the one topic-duplicate):

| # | Ticket ID | Requester employee ID (debug login) | Topic | Has a KB doc with the answer? |
|---|-----------|--------------------------------------|-------|-------------------------------|
| 1 | **1457** | 2569 | VPN access failure after password reset | Yes -- doc 552 |
| 2 | **1458** | 2570 | Facilities (coffee machine) -- no IT KB coverage at all | No (by design -- tests honest "not found") |
| 3 | **1459** | 2571 | Access/entitlement request (VPN for new project) | No KB doc; needs an entitlement check |
| 4 | **1460** | 2572 | Hardware (printer), ticket reopened -- needs history awareness | No KB doc; needs audit/history check |
| 5 | **1461** | 2573 | Account lockout (email), recovery-address policy | Yes -- doc 553 |

Why this mix is defensible, not artificially easy or hard for either side:
it spans the three distinct resolution shapes a real IT ticket actually
needs (pure KB-article lookup, "no answer exists, say so honestly,"
underlying-fact lookup that isn't KB content at all), matches exactly the
corpus Task 4 already treats as this project's canonical "realistic
diagnostic scenario" set, and was authored by this project itself (not by
this prep task) with `human_review_status: approved` in
`eval/golden/diagnostic_v1.jsonl` -- it was not picked to make either
condition look better.

## 2. Start/stop rules

- **Start**: the timer starts the moment the human is shown ticket N's
  subject + description (§3) with zero prior context of its content (they
  should not read ahead to ticket N+1 before finishing N -- work them in
  order 1->5, one at a time).
- **Stop**: the timer stops the instant the human has composed a final
  answer/resolution they would actually be willing to send to the
  requester (written down, even informally -- not just "I think I know
  it," an actual composed sentence or two).
- Record wall-clock elapsed seconds per ticket, plus the composed answer
  text, in a results file (template in §7).
- No AI assistance of any kind during this timed window: no `/chat`, no
  clicking "Diagnose" or "Summarize" on the ticket page, no asking any LLM
  (including this agent) for the answer. The only permitted tools: the
  real ticket detail page, `kb_export_for_manual_lookup.txt`, and for
  tickets #1459/#1460 only, the one specific read-only SQL query given in
  §4 for that ticket.

## 3. The five tickets' real subject + description (read these to start each timer)

Recorded directly here because, per finding 0.4 above, no live UI/API path
exists to show `TicketMessage.body` today.

**Ticket #1457** -- subject: "VPN connection fails after password reset"
> I reset my corporate password yesterday and now my VPN client refuses to
> connect -- it just hangs on 'authenticating' and then times out.

**Ticket #1458** -- subject: "Office coffee machine on the 3rd floor is broken and leaking"
> The coffee machine near the east stairwell is leaking water onto the
> floor and won't brew.

**Ticket #1459** -- subject: "Please grant me VPN access for a new project"
> I'm starting on the Falcon project next week and need VPN access to the
> staging environment. Can you check what access I currently have on file
> and tell me specifically what I'd need granted for this?

**Ticket #1460** -- subject: "Printer firmware update issue reopened -- still failing after first fix attempt"
> This printer firmware update issue was marked resolved yesterday but the
> same failure is happening again this morning. Please look at this
> ticket's own history -- has it really already been through a fix attempt
> before today?

**Ticket #1461** -- subject: "Locked out of corporate email inbox -- recovery message not arriving"
> I'm locked out of my corporate email inbox after too many failed login
> attempts on my email account. I triggered an email account-recovery
> message but it never arrived in the inbox I can see.

## 4. Correctness threshold (defined per ticket, from real KB/DB content, before measuring)

- **#1457 -- PASS** if the answer identifies this as a VPN/network-access
  issue tied to the password reset and recommends clearing the VPN
  client's cached session/credentials before reconnecting (doc 552's
  content, in `kb_export_for_manual_lookup.txt`). FAIL if it misidentifies
  the issue or invents unrelated content.
- **#1458 -- PASS** if the answer honestly states this is outside IT's
  knowledge base / is a facilities issue and should be routed to
  facilities, rather than fabricating an IT policy for it. FAIL if it
  invents an IT-sounding "fix" for a leaking coffee machine.
- **#1459 -- PASS** if the answer (a) correctly reports the employee
  currently has NO VPN entitlement on file (verify via:
  `docker exec resolvegrid-postgres bash -c 'psql -U "$POSTGRES_USER" -d
  "$POSTGRES_DB" -c "SELECT * FROM employee_entitlement WHERE
  employee_id = 2571;"'` -- real result as of prep time: 0 rows) and (b)
  recommends requesting/granting VPN access through the normal approval
  flow, without ever claiming access has already been granted. FAIL if it
  claims access was already granted, or gives no actionable
  recommendation.
- **#1460 -- PASS** if the answer reflects that this is a RECURRING/
  reopened issue, not a fresh first-time report (verify via:
  `docker exec resolvegrid-postgres bash -c 'psql -U "$POSTGRES_USER" -d
  "$POSTGRES_DB" -c "SELECT action, before_json, after_json, created_at
  FROM audit_log WHERE entity_type='"'"'ticket'"'"' AND entity_id=1460;"'`
  -- real result as of prep time: one `ticket.reopen` row,
  resolved->open). FAIL if it treats the ticket as brand-new with no
  acknowledgment of the prior resolve-then-reopen history.
- **#1461 -- PASS** if the answer identifies this as an email-inbox-
  lockout/account-recovery issue (NOT VPN) and is consistent with doc
  553's real recovery-address behavior (the recovery message goes to a
  separate recovery address on file, never the locked inbox itself; check
  that address). FAIL if it misidentifies this as a VPN/network issue or
  gives advice inconsistent with that policy.

A human grader (the same operator, after all 5 are done, or a second
reviewer) marks each composed answer PASS/FAIL against the rule above
before computing any aggregate statistic.

## 5. Sample size and why it's small (disclosed, not hidden)

n=5. This is the ENTIRE available real, human-reviewed, KB-matched
scenario corpus in this project as of 2026-11-11 (see §0.1-0.2) -- there
is no larger pool being down-sampled from. A single human operator has
limited available time for this exercise; 5 paired before/after
observations is what a real, honest, non-fabricated benchmark run in this
environment can produce. No percentile statistic (p50/p95) is meaningful
at n=5 -- report the raw 5 times, their mean, and the min/max, not a
percentile.

## 6. Failure treatment

If the human cannot arrive at a correct answer manually at all for a given
ticket (exhausts the KB export / SQL lookup and still can't produce a
PASS-grade answer), that is recorded as a REAL data point: the actual
elapsed wall-clock time until they give up and declare "I can't resolve
this manually," explicitly marked FAIL in the correctness column. It is
NOT excluded from the sample, NOT replaced, and NOT treated as missing
data. A long failure time is itself informative (the true cost of a
manual miss) and must be reported honestly in the final comparison, even
if it makes the manual condition look slower or the AI-assist win look
larger than a "best case" manual number would.

## 7. Documented bias and limitations (stated plainly, not resolved for the reader)

- **Single, non-blinded, codebase-familiar operator.** The person running
  the manual condition (the project's own operator) already has deep
  familiarity with this exact codebase, its KB corpus, and quite possibly
  the specific wording of these exact golden fixtures (they were
  authored as part of this same project, in an earlier phase). A genuinely
  new IT analyst, unfamiliar with Kestrel Corp's policies, would almost
  certainly take longer on every ticket. This plausibly biases the
  measured manual times FASTER than a true new-analyst baseline --
  which would make the AI-assist win LOOK SMALLER than it might be for a
  real new hire. It could also go the other way: if the operator is
  specifically rusty on which document number holds which fact (unlike an
  analyst who uses the real KB search tool daily), they could be slower
  than a trained analyst with proper tooling. Both directions are real
  possibilities; this protocol does not resolve which dominates, and the
  final report must say so rather than picking whichever framing is more
  flattering.
- **The "manual" condition's tooling is not representative of a real
  production IT-support tool.** Per §0.5-0.6, this product currently has
  no KB search UI and no entitlement/history-lookup UI at all. The
  manual-lookup proxy used here (a flat text file + two raw SQL queries)
  is almost certainly SLOWER and clunkier than what an actual IT-support
  analyst's real tooling (e.g. a proper wiki search, a ticketing system's
  built-in history view) would give them. This likely biases manual times
  SLOWER than a fair "manual, but with decent tools" baseline would be --
  which would make the AI-assist win LOOK LARGER than it should. This
  pulls in the opposite direction from the first bullet; net effect on
  the final ~80%-reduction claim is genuinely uncertain and must be
  reported as such, not averaged away.
- **n=5, one operator, one sitting.** No inter-rater reliability check on
  the correctness grading (same person who did the lookup also grades
  it, per §4 -- a real conflict of interest, disclosed, not hidden,
  unless a second reviewer is available to re-grade from the same written
  threshold).
- **Not blinded to which condition comes first.** The human will do the
  manual run before the AI run (so their total unfamiliarity with these
  5 specific tickets' content is only genuinely preserved for the manual
  condition, as intended) -- but this also means by the time the AI-side
  run happens, the operator already knows each ticket's correct answer
  from doing it manually, which has no bearing on the AI condition's own
  wall-clock time (that's measured server-side, not timed by eye) but is
  worth noting for completeness.

## 8. What happens next (NOT run yet)

1. **Human action required**: open each of the 5 tickets in order (§9 has
   exact URLs/login steps), time yourself per §2's rules, write down your
   composed answer and elapsed seconds per ticket into
   `eval/results/phase14_manual_benchmark_v1/manual_run_results.csv`
   (template below). Do not use `/chat`, "Diagnose," or any AI tool.
2. Once that file has real numbers for all 5 tickets, a human or this
   agent grades each answer PASS/FAIL per §4's threshold.
3. **Only after step 2 is complete**, a clean, separate AI-assisted run
   executes `POST /tickets/{id}/diagnose` against these exact same 5
   ticket IDs (1457, 1458, 1459, 1460, 1461), capturing real server-side
   wall-clock latency from `Span`/`AgentRun` rows (same measurement
   convention as Task 2's load run) -- not interleaved with the manual
   run, to avoid any priming effect in either direction.
4. The paired comparison (manual seconds vs. AI seconds, per ticket, plus
   correctness for both) is written into a `comparison_report.md` in this
   same directory, and claim 2 is reported honestly in the final Task 6
   benchmark report -- met, partially met, or not met, with the real
   numbers and all of §6's limitations carried forward, not dropped.

`manual_run_results.csv` template (create this file when running the
human session):
```
ticket_id,start_time,stop_time,elapsed_seconds,composed_answer,pass_fail,notes
1457,,,,,,
1458,,,,,,
1459,,,,,,
1460,,,,,,
1461,,,,,,
```

## 9. Exact instructions and real working URLs for the human operator

**Environment status, confirmed live as of 2026-11-11** (see this agent's
own verification):
- Postgres, Redis, Ollama, LiteLLM, OTel collector, Prometheus, Grafana,
  Langfuse: all `docker ps` healthy.
- Next.js web dev server: already running, reachable at
  **http://localhost:3000**.
- FastAPI dev server: was NOT running; started via
  `uv run --package resolvegrid-api python scripts/run_api_dev.py
  --reload` and confirmed healthy at **http://127.0.0.1:8000/health** ->
  `200`. Keep this process running for the whole benchmark session (both
  the manual run and the later AI-assisted run need it).

**One-time setup per ticket** (the app has no real auth yet -- it uses a
temporary "debug employee id" box, per
`apps/web/lib/debug-principal.ts`): **the debug employee id is
per-ticket**, not global -- each of these 5 tickets is only viewable by
its own requester employee id (confirmed directly: even an "admin,
global"-role id from a DIFFERENT ticket got a real 403 trying to view
another ticket -- this codebase's authz scoping is narrower than the role
name implies, a real finding worth a footnote but not a blocker here).

For each ticket, before opening it:
1. Go to **http://localhost:3000/tickets/{ticket_id}**.
2. In the "Your employee id (temporary debug login)" box, enter that
   ticket's employee id from the table below, then reload if the page
   shows an authorization error.
3. Read the subject/status shown on the page, then read that ticket's
   description from §3 above (not shown on the live page -- see finding
   0.4), and start your timer at that moment.
4. Do NOT click "Diagnose" or "Summarize" during the timed manual
   condition.
5. Search `eval/results/phase14_manual_benchmark_v1/kb_export_for_manual_lookup.txt`
   (open it in a text editor, Ctrl+F) for the relevant policy. For ticket
   #1459 or #1460 only, also run that ticket's one specific SQL query
   from §4.
6. Stop the timer the moment you've composed your final answer; write it
   and the elapsed seconds into `manual_run_results.csv`.

| Ticket | URL | Debug employee id to enter |
|--------|-----|------------------------------|
| 1457 | http://localhost:3000/tickets/1457 | 2569 |
| 1458 | http://localhost:3000/tickets/1458 | 2570 |
| 1459 | http://localhost:3000/tickets/1459 | 2571 |
| 1460 | http://localhost:3000/tickets/1460 | 2572 |
| 1461 | http://localhost:3000/tickets/1461 | 2573 |

## 10. Reproducibility -- how the 5 benchmark tickets were created

One-off script (not committed to the repo -- lives in this session's
scratchpad, reproduced here verbatim for anyone re-running this protocol
from scratch), run via:
```
uv run --package resolvegrid-api python <script path>
```
calling, in order, the real, unmodified
`resolvegrid_api.diagnostic_eval._build_vpn_access_issue`,
`_build_no_kb_evidence_honest_abstention`,
`_build_entitlement_request_no_false_claim`,
`_build_audit_history_grounding`, and
`_build_email_password_reset_categorization`, each against a real
`resolvegrid_api.db.session_factory()` session, deliberately never
calling the `cleanup()` closure each one returns. Resulting real IDs
(confirmed against the live DB):

```
case_id,ticket_id,requester_employee_id
diagnostic.vpn_access_issue.001,1457,2569
diagnostic.no_kb_evidence_honest_abstention.001,1458,2570
diagnostic.entitlement_request_no_false_claim.001,1459,2571
diagnostic.audit_history_grounding.001,1460,2572
diagnostic.email_password_reset_categorization.001,1461,2573
```

Re-running the script again would create a SECOND independent set of 5
(each builder mints a fresh `uuid4` suffix) -- run it once per benchmark
attempt, not repeatedly.

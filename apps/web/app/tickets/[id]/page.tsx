"use client";

import { use, useEffect, useState } from "react";
import { getDebugEmployeeId, setDebugEmployeeId } from "@/lib/debug-principal";
import { apiFetch } from "@/lib/api";

interface TicketDetail {
  id: number;
  subject: string;
  status: string;
  priority: string;
  type: string;
  requester_id: number;
  assignee_id: number | null;
}

interface SummarizeResult {
  summary: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  estimated_cost_usd: number;
}

interface DiagnosisCitation {
  chunk_id: number;
  document_title: string;
}

interface DiagnosisResult {
  summary: string;
  citations: DiagnosisCitation[];
  tool_calls_made: string[];
  thread_id: string;
}

// Keep in sync with ALLOWED_TRANSITIONS in
// packages/contracts/src/resolvegrid_contracts/tickets.py -- the backend is
// the single source of truth and is the actual enforcement point; this map
// only controls which buttons render.
const NEXT_STATUS_OPTIONS: Record<string, string[]> = {
  open: ["in_progress", "closed"],
  in_progress: ["resolved", "open", "closed"],
  resolved: ["closed", "reopened"],
  closed: ["reopened"],
  reopened: ["in_progress", "closed"],
};

export default function TicketDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [employeeId, setEmployeeIdState] = useState(() => getDebugEmployeeId());
  const [ticket, setTicket] = useState<TicketDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [summarizing, setSummarizing] = useState(false);
  const [summarizeResult, setSummarizeResult] = useState<SummarizeResult | null>(null);
  const [diagnosing, setDiagnosing] = useState(false);
  const [diagnosisResult, setDiagnosisResult] = useState<DiagnosisResult | null>(null);
  const [diagnoseError, setDiagnoseError] = useState<string | null>(null);
  const [diagnoseElapsedSeconds, setDiagnoseElapsedSeconds] = useState(0);

  function handleEmployeeIdChange(value: string) {
    setEmployeeIdState(value);
    setDebugEmployeeId(value);
  }

  function load() {
    if (!employeeId) return;
    apiFetch(`/tickets/${id}`)
      .then(async (response) => {
        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          const message =
            typeof detail.detail === "string" ? detail.detail : JSON.stringify(detail.detail ?? "request failed");
          setError(`Error ${response.status}: ${message}`);
          return;
        }
        setError(null);
        setTicket(await response.json());
      })
      .catch(() => setError("Network error"));
  }

  useEffect(load, [employeeId, id]);

  // Code review follow-up: a real diagnostic run's internal ADK tool-
  // calling loop can take up to _MAX_LLM_CALLS=60 turns (ADR 0004's own
  // cited finding) -- 20-50+ real seconds, an order of magnitude longer
  // than `summarize()`'s single LLM call. A static "Diagnosing..." label
  // with nothing moving risks reading as a frozen page to a real analyst
  // during that whole window. This ticks a plain elapsed-seconds counter
  // once a second while `diagnosing` is true, resetting to 0 the moment a
  // new run starts (see `diagnose()` below) -- deliberately NOT a fake
  // progress bar (there is no real sub-step progress to report; ADK's own
  // Runner owns its internal loop as one opaque unit, same reason
  // `routers/tickets.py`'s `diagnose_ticket_endpoint` writes only one real
  // `Span` row rather than a fabricated per-tool-call breakdown), just an
  // honest "time elapsed so far" signal that something is actively
  // running.
  useEffect(() => {
    if (!diagnosing) return;
    const interval = setInterval(() => setDiagnoseElapsedSeconds((s) => s + 1), 1000);
    return () => clearInterval(interval);
  }, [diagnosing]);

  async function transition(toStatus: string) {
    setMessage(null);
    setError(null);
    let response: Response;
    try {
      response = await apiFetch(`/tickets/${id}/transition`, {
        method: "POST",
        body: JSON.stringify({ to_status: toStatus }),
      });
    } catch {
      setError("Network error: could not reach the API. Is it running?");
      return;
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const message =
        typeof detail.detail === "string" ? detail.detail : JSON.stringify(detail.detail ?? "request failed");
      setError(`Error ${response.status}: ${message}`);
      return;
    }
    setMessage(`Transitioned to ${toStatus}`);
    load();
  }

  async function summarize() {
    setError(null);
    setMessage(null);
    setSummarizeResult(null);
    setSummarizing(true);
    let response: Response;
    try {
      response = await apiFetch(`/tickets/${id}/summarize`, { method: "POST" });
    } catch {
      setError("Network error: could not reach the API. Is it running?");
      setSummarizing(false);
      return;
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const message =
        typeof detail.detail === "string" ? detail.detail : JSON.stringify(detail.detail ?? "request failed");
      setError(`Error ${response.status}: ${message}`);
      setSummarizing(false);
      return;
    }
    setSummarizeResult(await response.json());
    setSummarizing(false);
  }

  // Phase 13 Task 4: "Diagnose" -- analyst/approver/admin-only action
  // (`ticket.diagnose`, a new staff-only authz action), running the real
  // Google ADK diagnostic specialist via `POST /tickets/{id}/diagnose`.
  // No client-side role check hides this button: `/approvals`
  // (apps/web/app/approvals/page.tsx) is this codebase's existing
  // precedent for a staff-only action with ZERO frontend role-gating --
  // every page here relies on the backend's real authz check (a 403
  // surfaced through the same `setError`-style pattern already used
  // below), not a client-side guess at the viewer's roles. Matched here
  // rather than inventing a new, inconsistent convention.
  async function diagnose() {
    setDiagnoseError(null);
    setDiagnosisResult(null);
    setDiagnoseElapsedSeconds(0);
    setDiagnosing(true);
    let response: Response;
    try {
      response = await apiFetch(`/tickets/${id}/diagnose`, { method: "POST" });
    } catch {
      setDiagnoseError("Network error: could not reach the API. Is it running?");
      setDiagnosing(false);
      return;
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const message =
        typeof detail.detail === "string" ? detail.detail : JSON.stringify(detail.detail ?? "request failed");
      setDiagnoseError(`Error ${response.status}: ${message}`);
      setDiagnosing(false);
      return;
    }
    setDiagnosisResult(await response.json());
    setDiagnosing(false);
  }

  return (
    <main>
      <h1>Ticket #{id}</h1>
      <p>
        <label>
          Your employee id (temporary debug login):{" "}
          <input value={employeeId} onChange={(e) => handleEmployeeIdChange(e.target.value)} />
        </label>
      </p>
      {error && <p role="alert">{error}</p>}
      {message && <p role="status">{message}</p>}
      {ticket && (
        <>
          <p>Subject: {ticket.subject}</p>
          <p>Status: {ticket.status}</p>
          <p>Priority: {ticket.priority}</p>
          <div>
            {(NEXT_STATUS_OPTIONS[ticket.status] ?? []).map((next) => (
              <button key={next} onClick={() => transition(next)}>
                Move to {next}
              </button>
            ))}
          </div>
          <div>
            <button onClick={summarize} disabled={summarizing}>
              {summarizing ? "Summarizing..." : "Summarize"}
            </button>
            {summarizeResult && (
              <div role="status">
                <p>{summarizeResult.summary}</p>
                <p>
                  <small>
                    {summarizeResult.latency_ms}ms · {summarizeResult.input_tokens}→{summarizeResult.output_tokens}{" "}
                    tokens · ${summarizeResult.estimated_cost_usd.toFixed(4)}
                  </small>
                </p>
              </div>
            )}
          </div>
          <div>
            <button onClick={diagnose} disabled={diagnosing}>
              {diagnosing ? "Diagnosing..." : "Diagnose"}
            </button>
            {diagnosing && (
              // Real, non-mocked diagnostic runs can take 20-50+ seconds
              // (see the useEffect above) -- a cycling dot animation
              // (changes every second, so it's visibly moving, not a
              // static label) plus a real elapsed-seconds counter, so
              // this reads as "working" the whole time, not "stuck" after
              // the first couple of seconds a static disabled-button
              // label alone would read fine for.
              <p role="status" aria-live="polite">
                Investigating ticket{".".repeat((diagnoseElapsedSeconds % 3) + 1)} ({diagnoseElapsedSeconds}s elapsed)
              </p>
            )}
            {diagnoseError && <p role="alert">{diagnoseError}</p>}
            {diagnosisResult && (
              <div role="status">
                <p>{diagnosisResult.summary}</p>
                {diagnosisResult.citations.length > 0 && (
                  <ul aria-label="citations">
                    {diagnosisResult.citations.map((citation) => (
                      <li key={citation.chunk_id}>
                        [chunk:{citation.chunk_id}] {citation.document_title}
                      </li>
                    ))}
                  </ul>
                )}
                <p>
                  <small>Tools used: {diagnosisResult.tool_calls_made.join(", ") || "none"}</small>
                </p>
              </div>
            )}
          </div>
        </>
      )}
    </main>
  );
}

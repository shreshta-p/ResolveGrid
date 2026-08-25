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
        </>
      )}
    </main>
  );
}

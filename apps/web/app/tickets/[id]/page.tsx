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
        </>
      )}
    </main>
  );
}

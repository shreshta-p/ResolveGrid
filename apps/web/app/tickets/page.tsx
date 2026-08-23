"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getDebugEmployeeId, setDebugEmployeeId } from "@/lib/debug-principal";
import { apiFetch } from "@/lib/api";

interface TicketSummary {
  id: number;
  subject: string;
  status: string;
  priority: string;
  requester_id: number;
}

export default function TicketQueuePage() {
  const [employeeId, setEmployeeIdState] = useState(() => getDebugEmployeeId());
  const [tickets, setTickets] = useState<TicketSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  function handleEmployeeIdChange(value: string) {
    setEmployeeIdState(value);
    setDebugEmployeeId(value);
  }

  useEffect(() => {
    if (!employeeId) return;
    apiFetch("/tickets")
      .then(async (response) => {
        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          const message =
            typeof detail.detail === "string" ? detail.detail : JSON.stringify(detail.detail ?? "request failed");
          setError(`Error ${response.status}: ${message}`);
          return;
        }
        setError(null);
        setTickets(await response.json());
      })
      .catch(() => setError("Network error"));
  }, [employeeId]);

  return (
    <main>
      <h1>Ticket queue</h1>
      <p>
        <label>
          Your employee id (temporary debug login):{" "}
          <input value={employeeId} onChange={(e) => handleEmployeeIdChange(e.target.value)} />
        </label>
      </p>
      {error && <p role="alert">{error}</p>}
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Subject</th>
            <th>Status</th>
            <th>Priority</th>
          </tr>
        </thead>
        <tbody>
          {tickets.map((t) => (
            <tr key={t.id}>
              <td>
                <Link href={`/tickets/${t.id}`}>{t.id}</Link>
              </td>
              <td>{t.subject}</td>
              <td>{t.status}</td>
              <td>{t.priority}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}

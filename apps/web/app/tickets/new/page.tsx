"use client";

import { useState } from "react";
import { getDebugEmployeeId, setDebugEmployeeId } from "@/lib/debug-principal";
import { apiFetch } from "@/lib/api";

export default function NewTicketPage() {
  const [employeeId, setEmployeeIdState] = useState(() => getDebugEmployeeId());
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [type, setType] = useState("incident");
  const [queueId, setQueueId] = useState("1");
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleEmployeeIdChange(value: string) {
    setEmployeeIdState(value);
    setDebugEmployeeId(value);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    let response: Response;
    try {
      response = await apiFetch("/tickets", {
        method: "POST",
        body: JSON.stringify({ subject, body, type, queue_id: Number(queueId) }),
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
    const ticket = await response.json();
    setResult(`Ticket #${ticket.id} created (status: ${ticket.status})`);
    setSubject("");
    setBody("");
  }

  return (
    <main>
      <h1>Submit a ticket</h1>
      <p>
        <label>
          Your employee id (temporary debug login):{" "}
          <input value={employeeId} onChange={(e) => handleEmployeeIdChange(e.target.value)} />
        </label>
      </p>
      <form onSubmit={handleSubmit}>
        <p>
          <label>
            Subject:{" "}
            <input value={subject} onChange={(e) => setSubject(e.target.value)} required />
          </label>
        </p>
        <p>
          <label>
            Description:{" "}
            <textarea value={body} onChange={(e) => setBody(e.target.value)} required />
          </label>
        </p>
        <p>
          <label>
            Type:{" "}
            <select value={type} onChange={(e) => setType(e.target.value)}>
              <option value="incident">Incident</option>
              <option value="service_request">Service Request</option>
            </select>
          </label>
        </p>
        <p>
          <label>
            Queue ID:{" "}
            <input value={queueId} onChange={(e) => setQueueId(e.target.value)} required />
          </label>
        </p>
        <button type="submit">Submit ticket</button>
      </form>
      {result && <p role="status">{result}</p>}
      {error && <p role="alert">{error}</p>}
    </main>
  );
}

"use client";

import { useState } from "react";
import { getDebugEmployeeId, setDebugEmployeeId } from "@/lib/debug-principal";
import { apiFetch } from "@/lib/api";

// Deliberately a single tiny form for the one mutating tool this phase
// registers (grant_vpn_access), not a general tool browser -- the
// two-tool registry (packages/contracts/src/resolvegrid_contracts/tools.py)
// doesn't warrant more, per the phase 9 plan doc's own instruction. Its
// only purpose is to give the /approvals page something real to approve.
interface InvokeResult {
  status: string;
  approval_request_id?: number;
  thread_id?: string;
  tool_name?: string;
  output?: unknown;
}

export default function ToolsPage() {
  const [employeeId, setEmployeeIdState] = useState(() => getDebugEmployeeId());
  const [targetEmployeeId, setTargetEmployeeId] = useState("");
  const [justification, setJustification] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<InvokeResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleEmployeeIdChange(value: string) {
    setEmployeeIdState(value);
    setDebugEmployeeId(value);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setSubmitting(true);
    let response: Response;
    try {
      response = await apiFetch("/tools/grant_vpn_access/invoke", {
        method: "POST",
        body: JSON.stringify({
          params: { employee_id: Number(targetEmployeeId), justification },
        }),
      });
    } catch {
      setError("Network error: could not reach the API. Is it running?");
      setSubmitting(false);
      return;
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const message =
        typeof detail.detail === "string" ? detail.detail : JSON.stringify(detail.detail ?? "request failed");
      setError(`Error ${response.status}: ${message}`);
      setSubmitting(false);
      return;
    }
    setResult(await response.json());
    setSubmitting(false);
  }

  return (
    <main>
      <h1>Invoke a tool: Grant VPN access</h1>
      <p>
        <label>
          Your employee id (temporary debug login):{" "}
          <input value={employeeId} onChange={(e) => handleEmployeeIdChange(e.target.value)} />
        </label>
      </p>
      <form onSubmit={handleSubmit}>
        <p>
          <label>
            Target employee id:{" "}
            <input value={targetEmployeeId} onChange={(e) => setTargetEmployeeId(e.target.value)} required />
          </label>
        </p>
        <p>
          <label>
            Justification:{" "}
            <input value={justification} onChange={(e) => setJustification(e.target.value)} required />
          </label>
        </p>
        <button type="submit" disabled={submitting}>
          {submitting ? "Submitting..." : "Request VPN access grant"}
        </button>
      </form>
      {result && (
        <div role="status">
          {result.status === "pending_approval" ? (
            <p>
              Pending approval. approval_request_id: {result.approval_request_id}, thread_id: {result.thread_id}.
              An approver must review this on the Approvals page before it takes effect.
            </p>
          ) : (
            <pre>{JSON.stringify(result, null, 2)}</pre>
          )}
        </div>
      )}
      {error && <p role="alert">{error}</p>}
    </main>
  );
}

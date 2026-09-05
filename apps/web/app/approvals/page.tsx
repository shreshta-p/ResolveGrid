"use client";

import { useCallback, useEffect, useState } from "react";
import { getDebugEmployeeId, setDebugEmployeeId } from "@/lib/debug-principal";
import { apiFetch } from "@/lib/api";

interface ApprovalRequestItem {
  id: number;
  action_type: string;
  action_params: Record<string, unknown>;
  risk_context: string | null;
  bound_evidence_refs: unknown[] | null;
  requested_by_id: number | null;
  status: string;
  expires_at: string | null;
  created_at: string | null;
}

interface ToolInvocationResult {
  status: string;
  output: unknown;
  error: string | null;
}

interface DecideResult {
  approval_request_id: number;
  decision: string;
  status: string;
  tool_invocation_result: ToolInvocationResult | null;
  resume_error?: string;
}

export default function ApprovalsPage() {
  const [employeeId, setEmployeeIdState] = useState(() => getDebugEmployeeId());
  const [requests, setRequests] = useState<ApprovalRequestItem[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [comments, setComments] = useState<Record<number, string>>({});
  const [validationErrors, setValidationErrors] = useState<Record<number, string>>({});
  const [decideErrors, setDecideErrors] = useState<Record<number, string>>({});
  const [decideResults, setDecideResults] = useState<Record<number, DecideResult>>({});
  const [submitting, setSubmitting] = useState<Record<number, boolean>>({});

  function handleEmployeeIdChange(value: string) {
    setEmployeeIdState(value);
    setDebugEmployeeId(value);
  }

  // A `.then()` chain, not `async`/`await` -- mirrors
  // `apps/web/app/tickets/page.tsx`'s established pattern for an effect-
  // driven fetch-on-mount: every setState call below happens inside a
  // `.then()`/`.catch()` callback (the promise resolves asynchronously,
  // after the effect that started it has already returned), never
  // synchronously within the effect's own call stack -- which is what
  // `react-hooks/set-state-in-effect` actually checks for.
  const loadRequests = useCallback(() => {
    return apiFetch("/approvals")
      .then(async (response) => {
        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          const message =
            typeof detail.detail === "string" ? detail.detail : JSON.stringify(detail.detail ?? "request failed");
          setListError(`Error ${response.status}: ${message}`);
          return;
        }
        setListError(null);
        setRequests(await response.json());
      })
      .catch(() => setListError("Network error: could not reach the API. Is it running?"));
  }, []);

  useEffect(() => {
    loadRequests();
  }, [loadRequests]);

  async function handleDecide(id: number, decision: "approved" | "rejected") {
    const comment = comments[id] ?? "";
    if (decision === "rejected" && comment.trim() === "") {
      setValidationErrors((prev) => ({ ...prev, [id]: "A comment is required when rejecting." }));
      return;
    }
    setValidationErrors((prev) => ({ ...prev, [id]: "" }));
    setDecideErrors((prev) => ({ ...prev, [id]: "" }));
    setSubmitting((prev) => ({ ...prev, [id]: true }));

    let response: Response;
    try {
      response = await apiFetch(`/approvals/${id}/decide`, {
        method: "POST",
        body: JSON.stringify({ decision, comment: comment.trim() === "" ? null : comment }),
      });
    } catch {
      setDecideErrors((prev) => ({ ...prev, [id]: "Network error: could not reach the API. Is it running?" }));
      setSubmitting((prev) => ({ ...prev, [id]: false }));
      return;
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const message =
        typeof detail.detail === "string" ? detail.detail : JSON.stringify(detail.detail ?? "request failed");
      setDecideErrors((prev) => ({ ...prev, [id]: `Error ${response.status}: ${message}` }));
      setSubmitting((prev) => ({ ...prev, [id]: false }));
      return;
    }
    const result: DecideResult = await response.json();
    setDecideResults((prev) => ({ ...prev, [id]: result }));
    setSubmitting((prev) => ({ ...prev, [id]: false }));
    await loadRequests();
  }

  return (
    <main>
      <h1>Approvals</h1>
      <p>
        <label>
          Your employee id (temporary debug login):{" "}
          <input value={employeeId} onChange={(e) => handleEmployeeIdChange(e.target.value)} />
        </label>
      </p>
      <p>
        <button type="button" onClick={loadRequests}>
          Refresh
        </button>
      </p>
      {listError && <p role="alert">{listError}</p>}
      {!listError && requests.length === 0 && <p>No pending approval requests.</p>}
      <ul>
        {requests.map((req) => (
          <li key={req.id}>
            <p>
              <strong>Request #{req.id}</strong> &mdash; {req.action_type} (status: {req.status})
            </p>
            <pre>{JSON.stringify(req.action_params, null, 2)}</pre>
            <p>Risk context: {req.risk_context ?? "(none)"}</p>
            {req.bound_evidence_refs && req.bound_evidence_refs.length > 0 && (
              <p>Evidence refs: {JSON.stringify(req.bound_evidence_refs)}</p>
            )}
            <p>Requested by employee #{req.requested_by_id ?? "(unknown)"}</p>
            <p>Expires at: {req.expires_at ?? "(unknown)"}</p>
            <p>
              <label>
                Comment (required to reject, optional to approve):{" "}
                <input
                  value={comments[req.id] ?? ""}
                  onChange={(e) => setComments((prev) => ({ ...prev, [req.id]: e.target.value }))}
                />
              </label>
            </p>
            {validationErrors[req.id] && <p role="alert">{validationErrors[req.id]}</p>}
            <p>
              <button type="button" disabled={submitting[req.id]} onClick={() => handleDecide(req.id, "approved")}>
                Approve
              </button>{" "}
              <button type="button" disabled={submitting[req.id]} onClick={() => handleDecide(req.id, "rejected")}>
                Reject
              </button>
            </p>
            {decideErrors[req.id] && <p role="alert">{decideErrors[req.id]}</p>}
            {decideResults[req.id] && (
              <div role="status">
                <p>
                  Decision recorded: {decideResults[req.id].decision} (request status:{" "}
                  {decideResults[req.id].status})
                </p>
                {decideResults[req.id].resume_error && <p role="alert">{decideResults[req.id].resume_error}</p>}
                {decideResults[req.id].tool_invocation_result && (
                  <pre>{JSON.stringify(decideResults[req.id].tool_invocation_result, null, 2)}</pre>
                )}
              </div>
            )}
          </li>
        ))}
      </ul>
    </main>
  );
}

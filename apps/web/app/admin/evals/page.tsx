"use client";

import { useCallback, useEffect, useState } from "react";
import { getDebugEmployeeId, setDebugEmployeeId } from "@/lib/debug-principal";
import { apiFetch } from "@/lib/api";

// Phase 10 Task 8: minimal admin dashboard reading real EvalRun/EvalCaseResult
// data from `GET /evals/runs` / `GET /evals/runs/{id}` -- matching
// `apps/web/app/approvals/page.tsx`'s established minimal style (plain
// HTML/CSS, no charting library) for a first cut; Phase 14 is where this
// gets visually polished, per the plan doc's own note.

interface DimensionSummary {
  passed_count: number;
  total_count: number;
  pass_rate: number | null;
}

interface RunSummary {
  dimensions: Record<string, DimensionSummary>;
  overall_passed_count: number;
  overall_total_count: number;
  overall_pass_rate: number | null;
}

interface EvalRunItem {
  id: number;
  dataset_version: string;
  prompt_version: string;
  workflow_version: string;
  retriever_version: string;
  reranker_version: string;
  embeddings_version: string;
  generation_model_version: string;
  judge_version: string;
  tool_schema_version: string;
  git_commit: string;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  summary: RunSummary;
}

interface EvalCaseResultItem {
  id: number;
  case_id: string;
  dimension: string;
  passed: boolean;
  score: number | null;
  grader_type: string;
  details: Record<string, unknown> | null;
  created_at: string | null;
}

interface EvalRunDetail extends EvalRunItem {
  case_results: EvalCaseResultItem[];
}

function formatPercent(rate: number | null): string {
  return rate === null ? "n/a" : `${Math.round(rate * 100)}%`;
}

// Plain HTML/CSS bar -- no charting library, per the plan doc's explicit
// instruction that this first cut doesn't need one.
function PassRateBar({ rate }: { rate: number | null }) {
  const percent = rate === null ? 0 : Math.round(rate * 100);
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem" }}>
      <span
        style={{
          display: "inline-block",
          width: "120px",
          height: "10px",
          background: "#ddd",
          border: "1px solid #999",
        }}
      >
        <span
          style={{
            display: "block",
            width: `${percent}%`,
            height: "100%",
            background: rate === null ? "#999" : rate >= 0.8 ? "#2e7d32" : rate >= 0.5 ? "#f9a825" : "#c62828",
          }}
        />
      </span>
      <span>{formatPercent(rate)}</span>
    </span>
  );
}

export default function AdminEvalsPage() {
  const [employeeId, setEmployeeIdState] = useState(() => getDebugEmployeeId());
  const [runs, setRuns] = useState<EvalRunItem[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [detail, setDetail] = useState<EvalRunDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  function handleEmployeeIdChange(value: string) {
    setEmployeeIdState(value);
    setDebugEmployeeId(value);
  }

  // Same `.then()`-chain convention `apps/web/app/approvals/page.tsx` uses
  // for its fetch-on-mount effect -- every setState call below happens
  // inside a `.then()`/`.catch()` callback, never synchronously within the
  // effect's own call stack.
  const loadRuns = useCallback(() => {
    return apiFetch("/evals/runs")
      .then(async (response) => {
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          const message =
            typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? "request failed");
          setListError(`Error ${response.status}: ${message}`);
          return;
        }
        setListError(null);
        setRuns(await response.json());
      })
      .catch(() => setListError("Network error: could not reach the API. Is it running?"));
  }, []);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  const loadDetail = useCallback((runId: number) => {
    setDetailLoading(true);
    setDetail(null);
    setDetailError(null);
    return apiFetch(`/evals/runs/${runId}`)
      .then(async (response) => {
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          const message =
            typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? "request failed");
          setDetailError(`Error ${response.status}: ${message}`);
          setDetailLoading(false);
          return;
        }
        setDetail(await response.json());
        setDetailLoading(false);
      })
      .catch(() => {
        setDetailError("Network error: could not reach the API. Is it running?");
        setDetailLoading(false);
      });
  }, []);

  function handleSelectRun(runId: number) {
    setSelectedRunId(runId);
    loadDetail(runId);
  }

  return (
    <main>
      <h1>Eval Runs</h1>
      <p>
        <label>
          Your employee id (temporary debug login):{" "}
          <input value={employeeId} onChange={(e) => handleEmployeeIdChange(e.target.value)} />
        </label>
      </p>
      <p>
        <button type="button" onClick={loadRuns}>
          Refresh
        </button>
      </p>
      {listError && <p role="alert">{listError}</p>}
      {!listError && runs.length === 0 && <p>No eval runs recorded yet.</p>}
      {runs.length > 0 && (
        <table border={1} cellPadding={4}>
          <thead>
            <tr>
              <th>Run</th>
              <th>Dataset version</th>
              <th>Status</th>
              <th>Started at</th>
              <th>Overall pass rate</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.id}>
                <td>#{run.id}</td>
                <td>{run.dataset_version}</td>
                <td>{run.status}</td>
                <td>{run.started_at ?? "(unknown)"}</td>
                <td>
                  <PassRateBar rate={run.summary.overall_pass_rate} /> (
                  {run.summary.overall_passed_count}/{run.summary.overall_total_count})
                </td>
                <td>
                  <button type="button" onClick={() => handleSelectRun(run.id)}>
                    {selectedRunId === run.id ? "Selected" : "View detail"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedRunId !== null && (
        <section>
          <h2>Run #{selectedRunId} detail</h2>
          {detailLoading && <p>Loading...</p>}
          {detailError && <p role="alert">{detailError}</p>}
          {detail && (
            <>
              <ul>
                <li>Git commit: {detail.git_commit}</li>
                <li>Prompt version: {detail.prompt_version}</li>
                <li>Workflow version: {detail.workflow_version}</li>
                <li>Retriever version: {detail.retriever_version}</li>
                <li>Reranker version: {detail.reranker_version}</li>
                <li>Embeddings version: {detail.embeddings_version}</li>
                <li>Generation model version: {detail.generation_model_version}</li>
                <li>Judge version: {detail.judge_version}</li>
                <li>Tool schema version: {detail.tool_schema_version}</li>
                <li>Completed at: {detail.completed_at ?? "(not completed)"}</li>
                {detail.error_message && <li role="alert">Error: {detail.error_message}</li>}
              </ul>

              <h3>Per-dimension pass rates</h3>
              <table border={1} cellPadding={4}>
                <thead>
                  <tr>
                    <th>Dimension</th>
                    <th>Pass rate</th>
                    <th>Passed / total</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(detail.summary.dimensions).map(([dimension, summary]) => (
                    <tr key={dimension}>
                      <td>{dimension}</td>
                      <td>
                        <PassRateBar rate={summary.pass_rate} />
                      </td>
                      <td>
                        {summary.passed_count}/{summary.total_count}
                      </td>
                    </tr>
                  ))}
                  <tr>
                    <td>
                      <strong>overall</strong>
                    </td>
                    <td>
                      <PassRateBar rate={detail.summary.overall_pass_rate} />
                    </td>
                    <td>
                      {detail.summary.overall_passed_count}/{detail.summary.overall_total_count}
                    </td>
                  </tr>
                </tbody>
              </table>

              <h3>Per-case results</h3>
              <table border={1} cellPadding={4}>
                <thead>
                  <tr>
                    <th>Case id</th>
                    <th>Dimension</th>
                    <th>Passed</th>
                    <th>Score</th>
                    <th>Grader type</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.case_results.map((caseResult) => (
                    <tr key={caseResult.id}>
                      <td>{caseResult.case_id}</td>
                      <td>{caseResult.dimension}</td>
                      <td>{caseResult.passed ? "pass" : "fail"}</td>
                      <td>{caseResult.score ?? "(n/a)"}</td>
                      <td>{caseResult.grader_type}</td>
                      <td>
                        {caseResult.details ? (
                          <details>
                            <summary>view</summary>
                            <pre>{JSON.stringify(caseResult.details, null, 2)}</pre>
                          </details>
                        ) : (
                          "(none)"
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </section>
      )}
    </main>
  );
}

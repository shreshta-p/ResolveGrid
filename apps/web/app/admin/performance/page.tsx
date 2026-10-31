"use client";

import { useCallback, useState } from "react";
import { getDebugEmployeeId, setDebugEmployeeId } from "@/lib/debug-principal";
import { apiFetch } from "@/lib/api";

// Phase 14 Task 3: minimal admin dashboard reading real AgentRun/Span/
// ModelCall aggregates from `GET /performance/metrics` -- matching
// `apps/web/app/admin/evals/page.tsx`'s established minimal style (plain
// HTML/CSS, no charting library, debug-employee-id-based auth). Visual
// polish is explicitly out of scope for this task (Phase 15's job per the
// plan doc); this page's only job is wiring the real endpoint correctly.
//
// Every number shown here traces to a real query against AgentRun/Span/
// ModelCall rows over whatever window is selected below -- there is no
// static/mock data path. Two real, honestly-reported instrumentation gaps
// (TTFT, retrieval-tier cache hit rate) are shown as explicit "not
// tracked" notices rather than silently omitted or faked.

type WindowMode = "window_minutes" | "load_test_run" | "explicit_range";

// `latency_ms.overall` comes straight from `agent_run.latency_ms_percentiles`
// (keys p50/p95/p99/mean, no n -- see `throughput.total_agent_run_count`
// for the count instead). `latency_ms.chat`/`latency_ms.diagnose` come from
// `agent_run_by_classification` (keys p50_ms/p95_ms/p99_ms/mean_ms/n) --
// two genuinely different real shapes from the shared aggregation core, not
// unified here so this page never silently reads a field that isn't there.
interface OverallLatencyPercentiles {
  p50?: number | null;
  p95?: number | null;
  p99?: number | null;
  mean?: number | null;
}

interface ClassifiedLatencyPercentiles {
  n: number;
  p50_ms: number | null;
  p95_ms: number | null;
  p99_ms: number | null;
  mean_ms: number | null;
}

interface MetricsResponse {
  window: {
    start: string;
    end: string;
    duration_seconds: number;
    mode: string;
    load_test_run_id?: number;
    load_test_run_started_at?: string;
    load_test_run_completed_at?: string;
    window_minutes?: number;
    buffer_seconds_applied: number;
  };
  throughput: {
    completed_agent_runs_per_sec: number | null;
    completed_count: number;
    total_agent_run_count: number;
  };
  latency_ms: {
    overall: OverallLatencyPercentiles;
    chat: ClassifiedLatencyPercentiles | null;
    diagnose: ClassifiedLatencyPercentiles | null;
  };
  error_rate: {
    agent_run_error_rate: number | null;
    error_count: number;
    total_count: number;
    status_counts: Record<string, number>;
  };
  cache: {
    hit_rate_overall: number | null;
    hit_count: number;
    total_count: number;
    hit_rate_by_tier: { exact: number | null; semantic: number | null; retrieval: number | null };
    known_gap_retrieval_tier: string;
  };
  cost: {
    total_model_call_cost_usd: number;
    cost_per_request_usd: number | null;
    cost_per_successful_task_usd: number | null;
    total_input_tokens: number;
    total_output_tokens: number;
    note: string;
  };
  ttft: { available: boolean; note: string };
  model_call_by_purpose: Array<{
    purpose: string;
    provider: string;
    model: string;
    n: number;
    input_tokens: number | null;
    output_tokens: number | null;
    cost_usd: number;
    mean_latency_ms: number | string;
    error_count: number;
  }>;
  span_by_stage: Record<string, { p50_ms: number | null; p95_ms: number | null; p99_ms: number | null; mean_ms: number | null; n: number }>;
}

function fmtMs(v: number | null | undefined): string {
  return v === null || v === undefined ? "n/a" : `${Math.round(v)} ms`;
}

function fmtPct(v: number | null | undefined): string {
  return v === null || v === undefined ? "n/a" : `${(v * 100).toFixed(1)}%`;
}

function fmtNum(v: number | null | undefined, digits = 3): string {
  return v === null || v === undefined ? "n/a" : v.toFixed(digits);
}

export default function AdminPerformancePage() {
  const [employeeId, setEmployeeIdState] = useState(() => getDebugEmployeeId());
  const [mode, setMode] = useState<WindowMode>("window_minutes");
  const [windowMinutes, setWindowMinutes] = useState("60");
  const [loadTestRunId, setLoadTestRunId] = useState("17");
  const [since, setSince] = useState("");
  const [until, setUntil] = useState("");
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function handleEmployeeIdChange(value: string) {
    setEmployeeIdState(value);
    setDebugEmployeeId(value);
  }

  const loadMetrics = useCallback(() => {
    const params = new URLSearchParams();
    if (mode === "window_minutes") {
      params.set("window_minutes", windowMinutes);
    } else if (mode === "load_test_run") {
      params.set("load_test_run_id", loadTestRunId);
    } else {
      params.set("since", since);
      params.set("until", until);
    }

    setLoading(true);
    setError(null);
    return apiFetch(`/performance/metrics?${params.toString()}`)
      .then(async (response) => {
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          const message =
            typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? "request failed");
          setError(`Error ${response.status}: ${message}`);
          setMetrics(null);
          setLoading(false);
          return;
        }
        setMetrics(await response.json());
        setLoading(false);
      })
      .catch(() => {
        setError("Network error: could not reach the API. Is it running?");
        setLoading(false);
      });
  }, [mode, windowMinutes, loadTestRunId, since, until]);

  return (
    <main>
      <h1>Admin Performance Dashboard</h1>
      <p>
        Real p50/p95/p99 latency, cache hit rate, token/cost totals, throughput, and error rate, computed on
        demand from <code>AgentRun</code>/<code>Span</code>/<code>ModelCall</code> rows over the window selected
        below. No static or mock data -- every number here traces to a live query.
      </p>

      <p>
        <label>
          Your employee id (temporary debug login):{" "}
          <input value={employeeId} onChange={(e) => handleEmployeeIdChange(e.target.value)} />
        </label>
      </p>

      <section>
        <h2>Window</h2>
        <p>
          <label>
            <input
              type="radio"
              name="mode"
              checked={mode === "window_minutes"}
              onChange={() => setMode("window_minutes")}
            />{" "}
            Trailing window (minutes):{" "}
          </label>
          <input
            value={windowMinutes}
            onChange={(e) => setWindowMinutes(e.target.value)}
            disabled={mode !== "window_minutes"}
            size={6}
          />
        </p>
        <p>
          <label>
            <input
              type="radio"
              name="mode"
              checked={mode === "load_test_run"}
              onChange={() => setMode("load_test_run")}
            />{" "}
            Phase 14 Task 2 load-test run id (e.g. 16 = full_system, 17 = baseline):{" "}
          </label>
          <input
            value={loadTestRunId}
            onChange={(e) => setLoadTestRunId(e.target.value)}
            disabled={mode !== "load_test_run"}
            size={6}
          />
        </p>
        <p>
          <label>
            <input
              type="radio"
              name="mode"
              checked={mode === "explicit_range"}
              onChange={() => setMode("explicit_range")}
            />{" "}
            Explicit range:{" "}
          </label>
          since{" "}
          <input
            value={since}
            onChange={(e) => setSince(e.target.value)}
            disabled={mode !== "explicit_range"}
            placeholder="2026-10-29T00:00:00"
            size={22}
          />{" "}
          until{" "}
          <input
            value={until}
            onChange={(e) => setUntil(e.target.value)}
            disabled={mode !== "explicit_range"}
            placeholder="2026-10-29T06:00:00"
            size={22}
          />
        </p>
        <p>
          <button type="button" onClick={loadMetrics} disabled={loading}>
            {loading ? "Loading..." : "Load metrics"}
          </button>
        </p>
      </section>

      {error && <p role="alert">{error}</p>}

      {metrics && (
        <>
          <section>
            <h2>Window actually queried</h2>
            <ul>
              <li>Start: {metrics.window.start}</li>
              <li>End: {metrics.window.end}</li>
              <li>Duration: {metrics.window.duration_seconds.toFixed(1)} s</li>
              <li>Mode: {metrics.window.mode}</li>
              {metrics.window.load_test_run_id !== undefined && (
                <li>Load test run id: {metrics.window.load_test_run_id}</li>
              )}
              {metrics.window.buffer_seconds_applied > 0 && (
                <li>Buffer applied each side: {metrics.window.buffer_seconds_applied} s</li>
              )}
            </ul>
          </section>

          <section>
            <h2>Throughput</h2>
            <ul>
              <li>
                Completed AgentRuns/sec:{" "}
                {metrics.throughput.completed_agent_runs_per_sec === null
                  ? "n/a"
                  : metrics.throughput.completed_agent_runs_per_sec.toFixed(4)}
              </li>
              <li>
                Completed / total AgentRuns: {metrics.throughput.completed_count} /{" "}
                {metrics.throughput.total_agent_run_count}
              </li>
            </ul>
          </section>

          <section>
            <h2>Latency (AgentRun wall-clock, created_at to completed_at)</h2>
            <table border={1} cellPadding={4}>
              <thead>
                <tr>
                  <th>Split</th>
                  <th>n</th>
                  <th>p50</th>
                  <th>p95</th>
                  <th>p99</th>
                  <th>mean</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>overall</td>
                  <td>{metrics.throughput.total_agent_run_count}</td>
                  <td>{fmtMs(metrics.latency_ms.overall.p50)}</td>
                  <td>{fmtMs(metrics.latency_ms.overall.p95)}</td>
                  <td>{fmtMs(metrics.latency_ms.overall.p99)}</td>
                  <td>{fmtMs(metrics.latency_ms.overall.mean)}</td>
                </tr>
                <tr>
                  <td>chat</td>
                  <td>{metrics.latency_ms.chat?.n ?? 0}</td>
                  <td>{fmtMs(metrics.latency_ms.chat?.p50_ms)}</td>
                  <td>{fmtMs(metrics.latency_ms.chat?.p95_ms)}</td>
                  <td>{fmtMs(metrics.latency_ms.chat?.p99_ms)}</td>
                  <td>{fmtMs(metrics.latency_ms.chat?.mean_ms)}</td>
                </tr>
                <tr>
                  <td>diagnose</td>
                  <td>{metrics.latency_ms.diagnose?.n ?? 0}</td>
                  <td>{fmtMs(metrics.latency_ms.diagnose?.p50_ms)}</td>
                  <td>{fmtMs(metrics.latency_ms.diagnose?.p95_ms)}</td>
                  <td>{fmtMs(metrics.latency_ms.diagnose?.p99_ms)}</td>
                  <td>{fmtMs(metrics.latency_ms.diagnose?.mean_ms)}</td>
                </tr>
              </tbody>
            </table>
            <p>
              <strong>TTFT (time to first token):</strong> {metrics.ttft.available ? "available" : "not tracked"} --{" "}
              {metrics.ttft.note}
            </p>
          </section>

          <section>
            <h2>Error rate</h2>
            <ul>
              <li>AgentRun error rate: {fmtPct(metrics.error_rate.agent_run_error_rate)}</li>
              <li>
                Error / total: {metrics.error_rate.error_count} / {metrics.error_rate.total_count}
              </li>
              <li>Status counts: {JSON.stringify(metrics.error_rate.status_counts)}</li>
            </ul>
          </section>

          <section>
            <h2>Cache hit rate</h2>
            <ul>
              <li>
                Overall: {fmtPct(metrics.cache.hit_rate_overall)} ({metrics.cache.hit_count}/{metrics.cache.total_count})
              </li>
              <li>Exact-match tier hits: {metrics.cache.hit_rate_by_tier.exact ?? 0}</li>
              <li>Semantic tier hits: {metrics.cache.hit_rate_by_tier.semantic ?? 0}</li>
              <li role="alert">
                Retrieval tier: not tracked -- {metrics.cache.known_gap_retrieval_tier}
              </li>
            </ul>
          </section>

          <section>
            <h2>Cost (token-based)</h2>
            <ul>
              <li>Total model-call cost (USD): {metrics.cost.total_model_call_cost_usd}</li>
              <li>Cost per request (AgentRun): {fmtNum(metrics.cost.cost_per_request_usd, 6)}</li>
              <li>Cost per successful task: {fmtNum(metrics.cost.cost_per_successful_task_usd, 6)}</li>
              <li>Total input tokens: {metrics.cost.total_input_tokens}</li>
              <li>Total output tokens: {metrics.cost.total_output_tokens}</li>
              <li role="alert">{metrics.cost.note}</li>
            </ul>
          </section>

          <section>
            <h2>Model calls by purpose</h2>
            <table border={1} cellPadding={4}>
              <thead>
                <tr>
                  <th>Purpose</th>
                  <th>Provider</th>
                  <th>Model</th>
                  <th>n</th>
                  <th>Input tokens</th>
                  <th>Output tokens</th>
                  <th>Cost USD</th>
                  <th>Mean latency ms</th>
                  <th>Errors</th>
                </tr>
              </thead>
              <tbody>
                {metrics.model_call_by_purpose.map((row, i) => (
                  <tr key={`${row.purpose}-${row.provider}-${row.model}-${i}`}>
                    <td>{row.purpose}</td>
                    <td>{row.provider}</td>
                    <td>{row.model}</td>
                    <td>{row.n}</td>
                    <td>{row.input_tokens ?? 0}</td>
                    <td>{row.output_tokens ?? 0}</td>
                    <td>{row.cost_usd}</td>
                    <td>{row.mean_latency_ms}</td>
                    <td>{row.error_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section>
            <h2>Spans by stage</h2>
            <table border={1} cellPadding={4}>
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>n</th>
                  <th>p50</th>
                  <th>p95</th>
                  <th>p99</th>
                  <th>mean</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.span_by_stage).map(([stage, row]) => (
                  <tr key={stage}>
                    <td>{stage}</td>
                    <td>{row.n}</td>
                    <td>{fmtMs(row.p50_ms)}</td>
                    <td>{fmtMs(row.p95_ms)}</td>
                    <td>{fmtMs(row.p99_ms)}</td>
                    <td>{fmtMs(row.mean_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}
    </main>
  );
}

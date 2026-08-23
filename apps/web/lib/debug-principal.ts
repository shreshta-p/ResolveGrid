// TEMPORARY: no real authentication exists yet (see docs/DECISION_LOG.md and
// apps/api/src/resolvegrid_api/deps.py). The user manually enters an
// employee id here, stored in localStorage, and it's attached as
// X-Debug-Employee-Id on every API call. Must be replaced with real
// authentication before any non-local deployment.
const STORAGE_KEY = "resolvegrid-debug-employee-id";

export function getDebugEmployeeId(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(STORAGE_KEY) ?? "";
}

export function setDebugEmployeeId(id: string): void {
  window.localStorage.setItem(STORAGE_KEY, id);
}

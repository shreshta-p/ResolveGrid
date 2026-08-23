import { getDebugEmployeeId } from "./debug-principal";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const employeeId = getDebugEmployeeId();
  const headers = new Headers(init?.headers);
  if (employeeId) {
    headers.set("X-Debug-Employee-Id", employeeId);
  }
  headers.set("Content-Type", "application/json");
  return fetch(`${API_BASE_URL}${path}`, { ...init, headers });
}

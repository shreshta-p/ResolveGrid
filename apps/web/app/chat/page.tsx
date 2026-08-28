"use client";

import { useState } from "react";
import { getDebugEmployeeId, setDebugEmployeeId } from "@/lib/debug-principal";
import { apiFetch } from "@/lib/api";

interface Citation {
  chunk_id: number;
  document_title: string;
}

interface ChatResult {
  answer: string;
  thread_id: string;
  citations: Citation[];
  caption: string;
}

export default function ChatPage() {
  const [employeeId, setEmployeeIdState] = useState(() => getDebugEmployeeId());
  const [message, setMessage] = useState("");
  const [thinking, setThinking] = useState(false);
  const [result, setResult] = useState<ChatResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleEmployeeIdChange(value: string) {
    setEmployeeIdState(value);
    setDebugEmployeeId(value);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setThinking(true);
    let response: Response;
    try {
      response = await apiFetch("/chat", {
        method: "POST",
        body: JSON.stringify({ message }),
      });
    } catch {
      setError("Network error: could not reach the API. Is it running?");
      setThinking(false);
      return;
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const message =
        typeof detail.detail === "string" ? detail.detail : JSON.stringify(detail.detail ?? "request failed");
      setError(`Error ${response.status}: ${message}`);
      setThinking(false);
      return;
    }
    setResult(await response.json());
    setThinking(false);
    setMessage("");
  }

  return (
    <main>
      <h1>Chat</h1>
      <p>
        <label>
          Your employee id (temporary debug login):{" "}
          <input value={employeeId} onChange={(e) => handleEmployeeIdChange(e.target.value)} />
        </label>
      </p>
      <form onSubmit={handleSubmit}>
        <p>
          <label>
            Ask a question:{" "}
            <textarea value={message} onChange={(e) => setMessage(e.target.value)} required />
          </label>
        </p>
        <button type="submit" disabled={thinking}>
          {thinking ? "Thinking..." : "Ask"}
        </button>
      </form>
      {result && (
        <div role="status">
          <p>{result.answer}</p>
          {result.citations.length > 0 && (
            <ul aria-label="citations">
              {result.citations.map((citation) => (
                <li key={citation.chunk_id}>
                  [chunk:{citation.chunk_id}] {citation.document_title}
                </li>
              ))}
            </ul>
          )}
          <p>
            <small>{result.caption}</small>
          </p>
        </div>
      )}
      {error && <p role="alert">{error}</p>}
    </main>
  );
}

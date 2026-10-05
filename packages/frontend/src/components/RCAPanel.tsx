import { useEffect, useState } from "react";
import type { MouseEvent } from "react";
import ReactMarkdown from "react-markdown";
import {
  createRcaSession,
  deleteRcaSession,
  listEvalHistory,
  listEvalQueries,
  listRcaMessages,
  listRcaSessions,
  sendRcaMessage,
} from "../api/client";
import type { EvalCycle, EvalQuery, RcaMessage, RcaSession } from "../api/client";

export function RCAPanel() {
  const [cycles, setCycles] = useState<EvalCycle[]>([]);
  const [queries, setQueries] = useState<EvalQuery[]>([]);
  const [selectedCycleId, setSelectedCycleId] = useState<number | null>(null);
  const [selectedQueryId, setSelectedQueryId] = useState<number | null>(null);
  const [sessions, setSessions] = useState<RcaSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null);
  const [messages, setMessages] = useState<RcaMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listEvalHistory()
      .then((result) => {
        setCycles(result);
        if (result.length > 0) setSelectedCycleId(result[0].id);
      })
      .catch((err) => setError((err as Error).message));
    listEvalQueries()
      .then((result) => {
        setQueries(result);
        if (result.length > 0) setSelectedQueryId(result[0].id);
      })
      .catch((err) => setError((err as Error).message));
    refreshSessions();
  }, []);

  function refreshSessions() {
    listRcaSessions()
      .then(setSessions)
      .catch((err) => setError((err as Error).message));
  }

  async function handleRunRca() {
    if (selectedCycleId === null || selectedQueryId === null) return;
    setRunning(true);
    setError(null);
    try {
      const result = await createRcaSession(selectedCycleId, selectedQueryId);
      setActiveSessionId(result.session_id);
      const history = await listRcaMessages(result.session_id);
      setMessages(history);
      refreshSessions();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRunning(false);
    }
  }

  async function handleSelectSession(sessionId: number) {
    setError(null);
    try {
      setActiveSessionId(sessionId);
      setMessages(await listRcaMessages(sessionId));
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function handleDeleteSession(sessionId: number, e: MouseEvent) {
    e.stopPropagation();
    setError(null);
    try {
      await deleteRcaSession(sessionId);
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
      }
      refreshSessions();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function handleSend() {
    if (activeSessionId === null || !draft.trim()) return;
    const question = draft;
    setDraft("");
    setMessages((prev) => [...prev, { role: "user", content: question, tool_trace: null }]);
    setRunning(true);
    setError(null);
    try {
      const reply = await sendRcaMessage(activeSessionId, question);
      setMessages((prev) => [...prev, reply]);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="rca-panel">
      <h2>RCA</h2>
      <p>Investigate why a specific eval run scored the way it did.</p>
      <div className="rca-layout">
        <aside className="rca-sidebar">
          <h3>Past sessions</h3>
          <ul>
            {sessions.map((s) => (
              <li key={s.id} className="rca-session-row">
                <button
                  className={s.id === activeSessionId ? "rca-session active" : "rca-session"}
                  onClick={() => handleSelectSession(s.id)}
                >
                  Cycle {s.cycle_id} - Q{s.query_id}
                  <span className="rca-session-tokens">
                    {s.total_input_tokens + s.total_output_tokens} tokens
                  </span>
                </button>
                <button
                  className="rca-session-delete"
                  onClick={(e) => handleDeleteSession(s.id, e)}
                  aria-label="Delete session"
                  title="Delete session"
                >
                  &times;
                </button>
              </li>
            ))}
            {sessions.length === 0 && <li>No past sessions yet.</li>}
          </ul>
        </aside>
        <div className="rca-main">
          <form className="config-form" onSubmit={(e) => e.preventDefault()}>
            <label>
              Eval Cycle
              <select
                value={selectedCycleId ?? ""}
                onChange={(e) => setSelectedCycleId(Number(e.target.value))}
              >
                {cycles.map((c) => (
                  <option key={c.id} value={c.id}>
                    Cycle {c.id} (config {c.config.id}{c.graph_augmented ? ", graph-augmented" : ""})
                  </option>
                ))}
              </select>
            </label>
            <label>
              Question
              <select
                value={selectedQueryId ?? ""}
                onChange={(e) => setSelectedQueryId(Number(e.target.value))}
              >
                {queries.map((q) => (
                  <option key={q.id} value={q.id}>
                    Q{q.id}: {q.query_text.slice(0, 60)}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" onClick={handleRunRca} disabled={running}>
              {running ? "Running..." : "Run RCA"}
            </button>
          </form>

          {error && <p className="error">{error}</p>}

          <div className="rca-chat">
            {messages.map((m, i) => (
              <div key={i} className={`rca-message rca-message-${m.role}`}>
                <strong>{m.role}</strong>
                {m.role === "assistant" ? (
                  <div className="rca-message-markdown">
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  </div>
                ) : (
                  <p>{m.content}</p>
                )}
                {m.tool_trace && m.tool_trace.length > 0 && (
                  <details>
                    <summary>Tool calls ({m.tool_trace.length})</summary>
                    <ul>
                      {m.tool_trace.map((t, j) => (
                        <li key={j}>
                          {t.tool}({JSON.stringify(t.args)}) -&gt; {t.result_summary}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            ))}
          </div>

          {activeSessionId !== null && (
            <div className="rca-input">
              <input
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                disabled={running}
                placeholder="Ask a follow-up..."
              />
              <button type="button" onClick={handleSend} disabled={running || !draft.trim()}>
                Send
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

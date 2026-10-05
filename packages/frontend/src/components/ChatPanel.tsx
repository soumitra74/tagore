import { useEffect, useState } from "react";
import { askQuestion, listConfigs } from "../api/client";
import type { ChatResponse, ExperimentConfig } from "../api/client";

const HISTORY_LIMIT = 10;

interface HistoryEntry {
  question: string;
  configId: number;
  result: ChatResponse;
}

export function ChatPanel() {
  const [configs, setConfigs] = useState<ExperimentConfig[]>([]);
  const [selectedConfigId, setSelectedConfigId] = useState<number | null>(null);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    listConfigs()
      .then((configsResult) => {
        setConfigs(configsResult);
        if (configsResult.length > 0) setSelectedConfigId(configsResult[0].id);
      })
      .catch((err) => setError((err as Error).message));
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim() || selectedConfigId === null) return;
    setLoading(true);
    setError(null);
    try {
      const response = await askQuestion(question, selectedConfigId);
      setResult(response);
      setHistory((prev) =>
        [{ question, configId: selectedConfigId, result: response }, ...prev].slice(
          0,
          HISTORY_LIMIT,
        ),
      );
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  function configLabel(configId: number): string {
    const c = configs.find((config) => config.id === configId);
    if (!c) return `config ${configId}`;
    return `${c.id} (${c.chunking_strategy}, ${c.chunk_size}/${c.chunk_overlap}, ${c.embedding_provider}: ${c.embedding_model})`;
  }

  return (
    <div className="chat-panel">
      <h2>Ask a question</h2>
      <form className="config-form" onSubmit={handleSubmit}>
        <label>
          Ingestion (config)
          <select
            value={selectedConfigId ?? ""}
            onChange={(e) => setSelectedConfigId(Number(e.target.value))}
          >
            {configs.map((c) => (
              <option key={c.id} value={c.id}>
                {configLabel(c.id)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Question
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask about the telemetry analysis..."
          />
        </label>
        <button type="submit" disabled={loading || selectedConfigId === null}>
          {loading ? "Asking..." : "Ask"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="answer">
          <p>{result.answer}</p>
          <ul>
            {result.sources.map((s, i) => (
              <li key={i}>
                {s.document} — page {s.page}: “{s.snippet}”
              </li>
            ))}
          </ul>
        </div>
      )}

      {history.length > 0 && (
        <div className="chat-history">
          <h2>Recent questions</h2>
          <table className="corpus-table">
            <thead>
              <tr>
                <th>Question</th>
                <th>Answer</th>
                <th>Ingestion</th>
              </tr>
            </thead>
            <tbody>
              {history.map((entry, i) => (
                <tr key={i}>
                  <td>{entry.question}</td>
                  <td>{entry.result.answer}</td>
                  <td>{configLabel(entry.configId)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

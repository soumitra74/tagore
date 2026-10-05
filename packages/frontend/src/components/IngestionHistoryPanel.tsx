import { useCallback, useEffect, useState } from "react";
import { listIngestionRuns } from "../api/client";
import type { IngestionRun } from "../api/client";

function formatTimestamp(isoString: string): string {
  return new Date(isoString).toLocaleString();
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function IngestionHistoryPanel() {
  const [runs, setRuns] = useState<IngestionRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setRuns(await listIngestionRuns());
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="ingestion-history-panel">
      <h2>Ingestion History</h2>
      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}
      {!loading && !error && runs.length === 0 && <p>No ingestion runs yet.</p>}
      {!loading && !error && runs.length > 0 && (
        <table className="corpus-table">
          <thead>
            <tr>
              <th>Ran at</th>
              <th>Config</th>
              <th>Chunking strategy</th>
              <th>Size / overlap</th>
              <th>Embedding</th>
              <th>Chunks</th>
              <th>Pages</th>
              <th>Time</th>
              <th>Errors</th>
              <th>Input size</th>
              <th>Vector DB size</th>
              <th>Results</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.id}>
                <td>{formatTimestamp(run.ran_at)}</td>
                <td>{run.config.id}</td>
                <td>{run.config.chunking_strategy}</td>
                <td>
                  {run.config.chunk_size} / {run.config.chunk_overlap}
                </td>
                <td>
                  {run.config.embedding_provider}: {run.config.embedding_model} (
                  {run.config.embedding_dim}d)
                </td>
                <td>{run.total_chunks}</td>
                <td>{run.total_pages}</td>
                <td>{(run.total_time_ms / 1000).toFixed(1)}s</td>
                <td>{run.error_count}</td>
                <td>{formatBytes(run.input_size_bytes)}</td>
                <td>{formatBytes(run.vector_db_size_bytes)}</td>
                <td>
                  <ul className="ingest-results">
                    {run.results.map((r) => (
                      <li key={r.filename}>
                        {r.filename}:{" "}
                        {r.error ? (
                          <span className="error">{r.error}</span>
                        ) : (
                          `${r.chunk_count} chunks, ${r.page_count} pages`
                        )}
                      </li>
                    ))}
                  </ul>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

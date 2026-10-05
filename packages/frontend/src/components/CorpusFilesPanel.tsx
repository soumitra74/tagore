import { useCallback, useEffect, useState } from "react";
import { ingestCorpusFile, isValidEmbeddingCombo, listCorpusFiles } from "../api/client";
import type { ExperimentConfig } from "../api/client";

interface CorpusFilesPanelProps {
  configs: ExperimentConfig[];
  onIngested: () => void;
}

export function CorpusFilesPanel({ configs, onIngested }: CorpusFilesPanelProps) {
  const [files, setFiles] = useState<string[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<Record<string, number | undefined>>({});
  const [status, setStatus] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setFiles(await listCorpusFiles());
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleIngest(filename: string) {
    const configId = selectedConfig[filename];
    const config = configId !== undefined ? configs.find((c) => c.id === configId) : undefined;
    if (
      config &&
      !isValidEmbeddingCombo(config.embedding_provider, config.embedding_model, config.embedding_dim)
    ) {
      setStatus((prev) => ({
        ...prev,
        [filename]:
          `Error: config ${config.id}'s ${config.embedding_model} on ${config.embedding_provider} ` +
          `does not produce ${config.embedding_dim}-dimensional vectors`,
      }));
      return;
    }
    setStatus((prev) => ({ ...prev, [filename]: "Ingesting..." }));
    try {
      const result = await ingestCorpusFile(filename, selectedConfig[filename]);
      setStatus((prev) => ({
        ...prev,
        [filename]:
          `Config ${result.config_id}: ${result.chunk_count} chunks, ${result.page_count} pages ` +
          `(${(result.metrics.total_time_ms / 1000).toFixed(1)}s)`,
      }));
      onIngested();
    } catch (err) {
      setStatus((prev) => ({ ...prev, [filename]: `Error: ${(err as Error).message}` }));
    }
  }

  return (
    <div className="corpus-files-panel">
      <h2>Corpus files</h2>
      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}
      {!loading && !error && files.length === 0 && (
        <p>No PDFs found in the corpus/ folder.</p>
      )}
      {!loading && !error && files.length > 0 && (
        <table className="corpus-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Config</th>
              <th></th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {files.map((filename) => (
              <tr key={filename}>
                <td>{filename}</td>
                <td>
                  <select
                    value={selectedConfig[filename] ?? ""}
                    onChange={(e) =>
                      setSelectedConfig((prev) => ({
                        ...prev,
                        [filename]: e.target.value ? Number(e.target.value) : undefined,
                      }))
                    }
                  >
                    <option value="">default</option>
                    {configs.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.id} ({c.chunking_strategy}, {c.chunk_size}/{c.chunk_overlap})
                      </option>
                    ))}
                  </select>
                </td>
                <td>
                  <button type="button" onClick={() => handleIngest(filename)}>
                    Ingest
                  </button>
                </td>
                <td>{status[filename]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

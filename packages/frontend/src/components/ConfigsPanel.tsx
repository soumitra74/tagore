import { useState } from "react";
import { EMBEDDING_MODEL_CATALOG, ingestConfigCorpus } from "../api/client";
import type {
  CorpusFileIngestResult,
  CreateConfigRequest,
  ExperimentConfig,
  IngestionMetrics,
} from "../api/client";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const CHUNKING_STRATEGIES = [
  {
    value: "sliding_window",
    label: "Sliding window",
    implemented: true,
    sizeLabel: "Chunk size (characters)",
    overlapLabel: "Chunk overlap (characters)",
  },
  {
    value: "fixed_token",
    label: "Fixed-token",
    implemented: true,
    sizeLabel: "Chunk size (tokens)",
    overlapLabel: "Chunk overlap (tokens)",
  },
  {
    value: "recursive_semantic",
    label: "Recursive / semantic",
    implemented: true,
    sizeLabel: "Chunk size (tokens)",
    overlapLabel: "Chunk overlap (tokens)",
  },
  {
    value: "sentence_window",
    label: "Sentence-window",
    implemented: true,
    sizeLabel: "Context window (sentences per side)",
    overlapLabel: "Chunk overlap (unused)",
  },
  {
    value: "document_structure_aware",
    label: "Document-structure-aware",
    implemented: true,
    sizeLabel: "Chunk size (unused)",
    overlapLabel: "Chunk overlap (unused)",
  },
] as const;

const EMBEDDING_PROVIDERS = ["local", "openai", "ollama"] as const;

const DEFAULT_FORM: CreateConfigRequest = {
  chunking_strategy: "sliding_window",
  chunk_size: 1000,
  chunk_overlap: 200,
  embedding_provider: "local",
  embedding_model: "BAAI/bge-base-en-v1.5",
};

interface ConfigsPanelProps {
  configs: ExperimentConfig[];
  onIngested: () => void;
}

export function ConfigsPanel({ configs, onIngested }: ConfigsPanelProps) {
  const [form, setForm] = useState<CreateConfigRequest>(DEFAULT_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<CorpusFileIngestResult[] | null>(null);
  const [metrics, setMetrics] = useState<IngestionMetrics | null>(null);

  const selectedStrategy =
    CHUNKING_STRATEGIES.find((s) => s.value === form.chunking_strategy) ?? CHUNKING_STRATEGIES[0];

  function updateField<K extends keyof CreateConfigRequest>(
    key: K,
    value: CreateConfigRequest[K],
  ) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function updateProvider(provider: string) {
    const firstModel = EMBEDDING_MODEL_CATALOG[provider]?.[0];
    setForm((prev) => ({
      ...prev,
      embedding_provider: provider,
      embedding_model: firstModel?.model ?? "",
    }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResults(null);
    setMetrics(null);
    setSubmitting(true);
    try {
      const response = await ingestConfigCorpus(form);
      setResults(response.results);
      setMetrics(response.metrics);
      onIngested();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="configs-panel">
      <h2>Experiment Configs</h2>
      {configs.length === 0 && <p>No configs yet.</p>}
      {configs.length > 0 && (
        <table className="corpus-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Chunking strategy</th>
              <th>Size / overlap</th>
              <th>Embedding</th>
            </tr>
          </thead>
          <tbody>
            {configs.map((c) => (
              <tr key={c.id}>
                <td>{c.id}</td>
                <td>{c.chunking_strategy}</td>
                <td>
                  {c.chunk_size} / {c.chunk_overlap}
                </td>
                <td>
                  {c.embedding_provider}: {c.embedding_model} ({c.embedding_dim}d)
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <form className="config-form" onSubmit={handleSubmit}>
        <label>
          Chunking strategy
          <select
            value={form.chunking_strategy}
            onChange={(e) => updateField("chunking_strategy", e.target.value)}
          >
            {CHUNKING_STRATEGIES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
                {s.implemented ? "" : " (not yet implemented)"}
              </option>
            ))}
          </select>
        </label>
        <label>
          {selectedStrategy.sizeLabel}
          <input
            type="number"
            min={1}
            value={form.chunk_size}
            onChange={(e) => updateField("chunk_size", Number(e.target.value))}
          />
        </label>
        <label>
          {selectedStrategy.overlapLabel}
          <input
            type="number"
            min={0}
            value={form.chunk_overlap}
            onChange={(e) => updateField("chunk_overlap", Number(e.target.value))}
          />
        </label>
        <label>
          Embedding provider
          <select value={form.embedding_provider} onChange={(e) => updateProvider(e.target.value)}>
            {EMBEDDING_PROVIDERS.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
        <label>
          Embedding model
          <select
            value={form.embedding_model}
            onChange={(e) => updateField("embedding_model", e.target.value)}
          >
            {(EMBEDDING_MODEL_CATALOG[form.embedding_provider] ?? []).map((m) => (
              <option key={m.model} value={m.model}>
                {m.model} ({m.dim}d)
              </option>
            ))}
          </select>
        </label>
        <button type="submit" disabled={submitting}>
          {submitting ? "Ingesting..." : "Ingest Corpus"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {metrics && (
        <table className="corpus-table">
          <thead>
            <tr>
              <th>Chunks</th>
              <th>Time</th>
              <th>Errors</th>
              <th>Input size</th>
              <th>Vector DB size</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>{metrics.chunks}</td>
              <td>{(metrics.total_time_ms / 1000).toFixed(1)}s</td>
              <td>{metrics.error_count}</td>
              <td>{formatBytes(metrics.input_size_bytes)}</td>
              <td>{formatBytes(metrics.vector_db_size_bytes)}</td>
            </tr>
          </tbody>
        </table>
      )}
      {results && (
        <ul className="ingest-results">
          {results.map((r) => (
            <li key={r.filename}>
              {r.filename}:{" "}
              {r.error ? (
                <span className="error">{r.error}</span>
              ) : (
                `${r.chunk_count} chunks`
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

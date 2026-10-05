import { useCallback, useEffect, useMemo, useState } from "react";
import { listEvalHistory } from "../api/client";
import type { EvalAggregates, EvalCycle } from "../api/client";
import {
  formatRetrievalMetric,
  retrievalMetricLabel,
  sortRetrievalMetricKeys,
} from "../retrievalMetrics";

type CompletedCycle = EvalCycle & { aggregates: EvalAggregates };

function hasAggregates(cycle: EvalCycle): cycle is CompletedCycle {
  return cycle.status === "completed" && cycle.aggregates != null;
}

// Every formatter below tolerates null/undefined and renders "-" instead of
// "NaN" -- a metric added after a cycle ran will be entirely absent from
// that cycle's stored aggregates JSON (frozen at whatever _summarize()
// produced at the time), not just zero, so this isn't a hypothetical edge
// case: every metric added has this gap for every older cycle.
type Num = number | null | undefined;

function formatPercent(value: Num): string {
  return value != null ? `${(value * 100).toFixed(0)}%` : "-";
}

function formatSeconds(msValue: Num): string {
  return msValue != null ? `${(msValue / 1000).toFixed(1)}s` : "-";
}

function formatFixed(value: Num, digits: number): string {
  return value != null ? value.toFixed(digits) : "-";
}

function formatCount(value: Num): string {
  return value != null ? `${value}` : "-";
}

function formatTimestamp(isoString: string): string {
  return new Date(isoString).toLocaleString();
}

interface MetricRow {
  key: string;
  label: string;
  format: (agg: EvalAggregates) => string;
  separatorAbove?: boolean;
}

// Single source of truth for both the on-screen table and the CSV export --
// each row formats its own value once, so the exported file always matches
// what's shown on screen.
// The dynamic retrieval-metric rows (see retrievalMetricRows below) are
// appended after these, under a separator.
const STATIC_METRIC_ROWS: MetricRow[] = [
  { key: "hit_rate", label: "Hit rate", format: (a) => formatPercent(a.hit_rate) },
  { key: "mean_mrr", label: "Mean MRR", format: (a) => formatFixed(a.mean_mrr, 2) },
  {
    key: "mean_citation_accuracy",
    label: "Mean citation accuracy",
    format: (a) => formatPercent(a.mean_citation_accuracy),
  },
  {
    key: "mean_citation_recall",
    label: "Mean citation recall",
    format: (a) => formatPercent(a.mean_citation_recall),
  },
  {
    key: "mean_correctness",
    label: "Mean correctness",
    format: (a) => formatPercent(a.mean_correctness),
  },
  {
    key: "mean_faithfulness",
    label: "Mean faithfulness",
    format: (a) => formatPercent(a.mean_faithfulness),
  },
  {
    key: "mean_answer_relevance",
    label: "Mean answer relevance",
    format: (a) => formatPercent(a.mean_answer_relevance),
  },
  {
    key: "mean_info_recall",
    label: "Mean info recall",
    format: (a) => formatPercent(a.mean_info_recall),
  },
  {
    key: "mean_citation_completeness",
    label: "Mean citation completeness",
    format: (a) => formatPercent(a.mean_citation_completeness),
  },
  {
    key: "mean_generation_latency_ms",
    label: "Mean generation latency",
    format: (a) => formatSeconds(a.mean_generation_latency_ms),
  },
  {
    key: "mean_rerank_latency_ms",
    label: "Mean rerank latency",
    format: (a) => formatSeconds(a.mean_rerank_latency_ms),
  },
  {
    key: "total_generation_input_tokens",
    label: "Total generation input tokens",
    format: (a) => formatCount(a.total_generation_input_tokens),
  },
  {
    key: "total_generation_output_tokens",
    label: "Total generation output tokens",
    format: (a) => formatCount(a.total_generation_output_tokens),
  },
  {
    key: "mean_eval_latency_ms",
    label: "Mean judge (eval) latency",
    format: (a) => formatSeconds(a.mean_eval_latency_ms),
  },
  {
    key: "total_eval_input_tokens",
    label: "Total judge (eval) input tokens",
    format: (a) => formatCount(a.total_eval_input_tokens),
  },
  {
    key: "total_eval_output_tokens",
    label: "Total judge (eval) output tokens",
    format: (a) => formatCount(a.total_eval_output_tokens),
  },
  { key: "total_llm_calls", label: "Total LLM calls", format: (a) => formatCount(a.total_llm_calls) },
  { key: "num_queries", label: "Queries evaluated", format: (a) => formatCount(a.num_queries) },
];

// The retrieval metrics (Recall@k, nDCG@k, ...) are a free-form keyed map
// (recall k-set is configurable), so rows are generated from whatever keys
// appear across the compared cycles' aggregates -- a cycle that lacks a key
// (older, or a different k-set) just shows "-".
// The default set is always shown (as "-" where absent) so the rows are
// visible even before any compared cycle has run with these metrics.
const DEFAULT_RETRIEVAL_METRIC_KEYS = ["recall@5", "recall@10", "recall@20", "recall@50", "ndcg@10"];

function retrievalMetricRows(cycles: CompletedCycle[]): MetricRow[] {
  const keys = new Set<string>(DEFAULT_RETRIEVAL_METRIC_KEYS);
  for (const cycle of cycles) {
    for (const key of Object.keys(cycle.aggregates.mean_retrieval_metrics ?? {})) {
      keys.add(key);
    }
  }
  return sortRetrievalMetricKeys(Array.from(keys)).map((key, i) => ({
    separatorAbove: i === 0,
    key: `mean_${key}`,
    label: `Mean ${retrievalMetricLabel(key)}`,
    format: (a: EvalAggregates) =>
      formatRetrievalMetric(key, a.mean_retrieval_metrics?.[key]),
  }));
}

function configLabel(cycle: EvalCycle): string {
  const { config } = cycle;
  const rerank = cycle.rerank_model
    ? `, rerank ${cycle.rerank_model.split("/").pop()} x${cycle.rerank_candidates}`
    : "";
  return `${config.chunking_strategy} (${config.chunk_size}/${config.chunk_overlap}), ${config.embedding_provider}: ${config.embedding_model}${rerank}`;
}

function csvEscape(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

function downloadCsv(filename: string, rows: string[][]) {
  const content = rows.map((row) => row.map(csvEscape).join(",")).join("\r\n");
  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function EvalComparePanel() {
  const [cycles, setCycles] = useState<EvalCycle[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const history = await listEvalHistory();
      setCycles(history);
      // history is ordered most-recent-first (started_at DESC), so the
      // first 4 completed cycles are the 4 most recent ones.
      const defaultSelected = history.filter(hasAggregates).slice(0, 4).map((c) => c.id);
      setSelected(new Set(defaultSelected));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const completedCycles = useMemo(() => cycles.filter(hasAggregates), [cycles]);
  const comparedCycles = useMemo(
    () => completedCycles.filter((c) => selected.has(c.id)),
    [completedCycles, selected],
  );
  const allMetricRows = useMemo(
    () => [
      ...STATIC_METRIC_ROWS,
      ...retrievalMetricRows(comparedCycles),
    ],
    [comparedCycles],
  );

  function toggleSelected(id: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function handleExportCsv() {
    const header = ["Metric", ...comparedCycles.map((c) => `Run #${c.id}`)];
    const rows = allMetricRows.map((row) => [
      row.label,
      ...comparedCycles.map((c) => row.format(c.aggregates)),
    ]);
    downloadCsv(`eval-comparison-${Date.now()}.csv`, [header, ...rows]);
  }

  return (
    <div className="eval-compare-panel">
      <h2>Compare Evals</h2>
      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}
      {!loading && !error && completedCycles.length === 0 && (
        <p>No fully completed eval runs yet.</p>
      )}
      {!loading && !error && completedCycles.length > 0 && (
        <>
          <ul className="run-picker">
            {completedCycles.map((cycle) => (
              <li key={cycle.id}>
                <label>
                  <input
                    type="checkbox"
                    checked={selected.has(cycle.id)}
                    onChange={() => toggleSelected(cycle.id)}
                  />
                  Run #{cycle.id} &mdash; {configLabel(cycle)} &mdash;{" "}
                  {formatTimestamp(cycle.completed_at ?? cycle.started_at)}
                </label>
              </li>
            ))}
          </ul>

          {comparedCycles.length === 0 ? (
            <p>Select at least one run to compare.</p>
          ) : (
            <>
              <button type="button" onClick={handleExportCsv}>
                Export CSV
              </button>
              <div className="table-scroll">
                <table className="corpus-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      {comparedCycles.map((cycle) => (
                        <th key={cycle.id}>Run #{cycle.id}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {allMetricRows.map((row) => (
                      <tr
                        key={row.key}
                        style={row.separatorAbove ? { borderTop: "2px solid currentColor" } : undefined}
                      >
                        <td>{row.label}</td>
                        {comparedCycles.map((cycle) => (
                          <td key={cycle.id}>{row.format(cycle.aggregates)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}

import { Fragment, useCallback, useEffect, useState } from "react";
import { deleteEvalCycle, listEvalHistory, purgeEvalCycle } from "../api/client";
import type { EvalCycle, EvalCycleQueryResult } from "../api/client";
import {
  formatRetrievalMetric,
  retrievalMetricLabel,
  sortRetrievalMetricKeys,
} from "../retrievalMetrics";

function formatTimestamp(isoString: string): string {
  return new Date(isoString).toLocaleString();
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

function formatPercentOrDash(value: number | null | undefined): string {
  return value != null ? formatPercent(value) : "-";
}

export function EvalHistoryPanel() {
  const [cycles, setCycles] = useState<EvalCycle[]>([]);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const [expandedQueries, setExpandedQueries] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [hidingId, setHidingId] = useState<number | null>(null);
  const [purgingId, setPurgingId] = useState<number | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setCycles(await listEvalHistory());
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  function toggleExpanded(id: number) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleQueryExpanded(key: string) {
    setExpandedQueries((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  async function handleHide(cycleId: number) {
    if (
      !window.confirm(
        `Hide eval run #${cycleId} from this list? The data stays in the database.`,
      )
    ) {
      return;
    }
    setHidingId(cycleId);
    setError(null);
    try {
      await deleteEvalCycle(cycleId);
      setCycles((prev) => prev.filter((c) => c.id !== cycleId));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setHidingId(null);
    }
  }

  async function handlePurge(cycleId: number) {
    if (
      !window.confirm(
        `Permanently delete eval run #${cycleId} and all its question results? This cannot be undone.`,
      )
    ) {
      return;
    }
    setPurgingId(cycleId);
    setError(null);
    try {
      await purgeEvalCycle(cycleId);
      setCycles((prev) => prev.filter((c) => c.id !== cycleId));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setPurgingId(null);
    }
  }

  return (
    <div className="eval-history-panel">
      <h2>Eval History</h2>
      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}
      {!loading && !error && cycles.length === 0 && <p>No eval runs yet.</p>}
      {!loading && !error && cycles.length > 0 && (
        <table className="corpus-table">
          <thead>
            <tr>
              <th></th>
              <th>Started</th>
              <th>Status</th>
              <th>Config</th>
              <th>Hit rate</th>
              <th>Mean correctness</th>
              <th>Mean faithfulness</th>
              <th>Mean generation latency</th>
              <th>Total generation tokens (in/out)</th>
              <th></th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {cycles.map((cycle) => (
              <Fragment key={cycle.id}>
                <tr>
                  <td>
                    <button
                      type="button"
                      className="expand-btn"
                      onClick={() => toggleExpanded(cycle.id)}
                      aria-expanded={expanded.has(cycle.id)}
                    >
                      {expanded.has(cycle.id) ? "-" : "+"}
                    </button>
                  </td>
                  <td>{formatTimestamp(cycle.started_at)}</td>
                  <td>
                    {cycle.status} ({cycle.per_query.length}/{cycle.total})
                  </td>
                  <td>
                    {cycle.config.id} ({cycle.config.chunking_strategy},{" "}
                    {cycle.config.chunk_size}/{cycle.config.chunk_overlap},{" "}
                    {cycle.config.embedding_provider}: {cycle.config.embedding_model}
                    {cycle.judge_provider
                      ? `, judge: ${cycle.judge_provider}: ${cycle.judge_model}`
                      : ""}
                    {cycle.graph_augmented ? ", graph-augmented" : ""}
                    {cycle.rerank_model
                      ? `, reranked (${cycle.rerank_model.split("/").pop()}, top ${cycle.rerank_candidates})`
                      : ""}
                    )
                  </td>
                  <td>{cycle.aggregates ? formatPercent(cycle.aggregates.hit_rate) : "-"}</td>
                  <td>
                    {cycle.aggregates ? formatPercent(cycle.aggregates.mean_correctness) : "-"}
                  </td>
                  <td>{formatPercentOrDash(cycle.aggregates?.mean_faithfulness)}</td>
                  <td>
                    {cycle.aggregates?.mean_generation_latency_ms != null
                      ? `${(cycle.aggregates.mean_generation_latency_ms / 1000).toFixed(1)}s`
                      : "-"}
                  </td>
                  <td>
                    {cycle.aggregates?.total_generation_input_tokens != null
                      ? `${cycle.aggregates.total_generation_input_tokens} / ${cycle.aggregates.total_generation_output_tokens}`
                      : "-"}
                  </td>
                  <td>
                    <button
                      type="button"
                      onClick={() => handleHide(cycle.id)}
                      disabled={hidingId === cycle.id || purgingId === cycle.id}
                    >
                      {hidingId === cycle.id ? "Hiding..." : "Hide"}
                    </button>
                  </td>
                  <td>
                    <button
                      type="button"
                      onClick={() => handlePurge(cycle.id)}
                      disabled={hidingId === cycle.id || purgingId === cycle.id}
                    >
                      {purgingId === cycle.id ? "Purging..." : "Delete"}
                    </button>
                  </td>
                </tr>
                {expanded.has(cycle.id) && (
                  <tr>
                    <td colSpan={11}>
                      <table className="corpus-table">
                        <thead>
                          <tr>
                            <th></th>
                            <th>Question</th>
                            <th>Hit</th>
                            <th>Correctness</th>
                            <th>Faithfulness</th>
                            <th>Generation latency</th>
                          </tr>
                        </thead>
                        <tbody>
                          {cycle.per_query.map((r) => {
                            const key = `${cycle.id}-${r.id}`;
                            return (
                              <EvalHistoryQueryRow
                                key={key}
                                r={r}
                                isOpen={expandedQueries.has(key)}
                                onToggle={() => toggleQueryExpanded(key)}
                              />
                            );
                          })}
                        </tbody>
                      </table>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function EvalHistoryQueryRow({
  r,
  isOpen,
  onToggle,
}: {
  r: EvalCycleQueryResult;
  isOpen: boolean;
  onToggle: () => void;
}) {
  return (
    <Fragment>
      <tr>
        <td>
          <button type="button" className="expand-btn" onClick={onToggle} aria-expanded={isOpen}>
            {isOpen ? "-" : "+"}
          </button>
        </td>
        <td>{r.question}</td>
        <td>{r.hit_rate ? "yes" : "no"}</td>
        <td>{formatPercent(r.correctness)}</td>
        <td>{formatPercentOrDash(r.faithfulness)}</td>
        <td>{(r.generation_latency_ms / 1000).toFixed(1)}s</td>
      </tr>
      {isOpen && (
        <tr>
          <td colSpan={6}>
            <div className="eval-detail">
              <div className="detail-item">
                <div className="detail-label">MRR</div>
                <div className="detail-value">{r.mrr.toFixed(2)}</div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Citation accuracy</div>
                <div className="detail-value">{formatPercent(r.citation_accuracy)}</div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Citation recall</div>
                <div className="detail-value">{formatPercentOrDash(r.citation_recall)}</div>
              </div>
              {sortRetrievalMetricKeys(Object.keys(r.retrieval_metrics ?? {})).map((key) => (
                <div className="detail-item" key={`retrieval-${key}`}>
                  <div className="detail-label">{retrievalMetricLabel(key)}</div>
                  <div className="detail-value">
                    {formatRetrievalMetric(key, r.retrieval_metrics?.[key])}
                  </div>
                </div>
              ))}
              {r.rerank_latency_ms != null && (
                <div className="detail-item">
                  <div className="detail-label">Rerank latency</div>
                  <div className="detail-value">{(r.rerank_latency_ms / 1000).toFixed(2)}s</div>
                </div>
              )}
              <div className="detail-item">
                <div className="detail-label">Answer relevance</div>
                <div className="detail-value">{formatPercentOrDash(r.answer_relevance)}</div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Info recall</div>
                <div className="detail-value">{formatPercentOrDash(r.info_recall)}</div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Citation completeness</div>
                <div className="detail-value">
                  {formatPercentOrDash(r.citation_completeness)}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Judge latency</div>
                <div className="detail-value">{(r.eval_latency_ms / 1000).toFixed(1)}s</div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Generation tokens (in/out)</div>
                <div className="detail-value">
                  {r.generation_input_tokens ?? "-"} / {r.generation_output_tokens ?? "-"}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Judge tokens (in/out)</div>
                <div className="detail-value">
                  {r.eval_input_tokens ?? "-"} / {r.eval_output_tokens ?? "-"}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-label">LLM calls</div>
                <div className="detail-value">{r.llm_call_count ?? "-"}</div>
              </div>
              <div className="detail-item detail-answer">
                <div className="detail-label">Generated answer</div>
                <div className="detail-value">{r.generated_answer}</div>
              </div>
            </div>
          </td>
        </tr>
      )}
    </Fragment>
  );
}

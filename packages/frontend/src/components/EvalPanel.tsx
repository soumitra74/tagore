import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_RERANK_CANDIDATES,
  RERANK_MODELS,
  getEvalJudge,
  getEvalRun,
  listConfigs,
  startEvalRun,
} from "../api/client";
import type { EvalJobStatus, EvalJudgeInfo, EvalQueryResult, ExperimentConfig } from "../api/client";
import { ChatPanel } from "./ChatPanel";

const POLL_INTERVAL_MS = 2000;

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

export function EvalPanel() {
  const [configs, setConfigs] = useState<ExperimentConfig[]>([]);
  const [selectedConfigId, setSelectedConfigId] = useState<number | null>(null);
  const [judge, setJudge] = useState<EvalJudgeInfo | null>(null);
  const [job, setJob] = useState<EvalJobStatus | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [graphAugmented, setGraphAugmented] = useState(false);
  const [topK, setTopK] = useState(""); // empty = server default
  const [rerankModel, setRerankModel] = useState(""); // empty = reranking off
  const [rerankCandidates, setRerankCandidates] = useState(DEFAULT_RERANK_CANDIDATES);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    listConfigs()
      .then((result) => {
        setConfigs(result);
        if (result.length > 0) setSelectedConfigId(result[0].id);
      })
      .catch((err) => setError((err as Error).message));
    getEvalJudge()
      .then(setJudge)
      .catch((err) => setError((err as Error).message));
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  function toggleExpanded(id: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function handleRunEval() {
    if (selectedConfigId === null) return;
    stopPolling();
    setStarting(true);
    setError(null);
    setJob(null);
    setExpanded(new Set());
    try {
      const { job_id } = await startEvalRun(
        selectedConfigId,
        topK ? Number(topK) : undefined,
        graphAugmented,
        rerankModel ? { model: rerankModel, candidates: rerankCandidates } : undefined,
      );
      const status = await getEvalRun(job_id);
      setJob(status);
      pollRef.current = setInterval(async () => {
        try {
          const polled = await getEvalRun(job_id);
          setJob(polled);
          if (polled.status !== "running") stopPolling();
        } catch (err) {
          setError((err as Error).message);
          stopPolling();
        }
      }, POLL_INTERVAL_MS);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setStarting(false);
    }
  }

  return (
    <div className="eval-panel">
      <h2>Run Eval Cycle</h2>
      <form
        className="config-form"
        onSubmit={(e) => {
          e.preventDefault();
          handleRunEval();
        }}
      >
        <label>
          Config
          <select
            value={selectedConfigId ?? ""}
            onChange={(e) => setSelectedConfigId(Number(e.target.value))}
          >
            {configs.map((c) => (
              <option key={c.id} value={c.id}>
                {c.id} ({c.chunking_strategy}, {c.chunk_size}/{c.chunk_overlap},{" "}
                {c.embedding_provider}: {c.embedding_model}
                {judge ? `, judge: ${judge.provider}: ${judge.model}` : ""})
              </option>
            ))}
          </select>
        </label>
        <label>
          <input
            type="checkbox"
            checked={graphAugmented}
            onChange={(e) => setGraphAugmented(e.target.checked)}
          />
          Graph-augmented retrieval (entity traversal)
        </label>
        <label>
          Top-K (generation context)
          <input
            type="number"
            min={1}
            max={30}
            placeholder="default"
            value={topK}
            onChange={(e) => setTopK(e.target.value)}
          />
        </label>
        <label>
          Reranker
          <select value={rerankModel} onChange={(e) => setRerankModel(e.target.value)}>
            <option value="">Off</option>
            {RERANK_MODELS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        {rerankModel && (
          <label>
            Rerank candidates
            <input
              type="number"
              min={1}
              max={50}
              value={rerankCandidates}
              onChange={(e) => setRerankCandidates(Number(e.target.value))}
            />
          </label>
        )}
        <button type="submit" disabled={starting || selectedConfigId === null}>
          {starting ? "Starting..." : "Run Eval"}
        </button>
      </form>
      {judge && (
        <p>
          Judge LLM: {judge.provider}: {judge.model} (
          {judge.combine_calls ? "combined judge call" : "separate judge calls"})
        </p>
      )}
      {error && <p className="error">{error}</p>}

      {job && (
        <div className="eval-progress">
          <p>
            Status: {job.status} ({job.per_query.length}/{job.total})
          </p>

          {job.aggregates && (
            <div className="kpi-strip">
              <div className="kpi-tile">
                <span className="kpi-label">Hit rate</span>
                <span className="kpi-value">{formatPercent(job.aggregates.hit_rate)}</span>
              </div>
              <div className="kpi-tile">
                <span className="kpi-label">Mean correctness</span>
                <span className="kpi-value">{formatPercent(job.aggregates.mean_correctness)}</span>
              </div>
              <div className="kpi-tile">
                <span className="kpi-label">Mean faithfulness</span>
                <span className="kpi-value">{formatPercent(job.aggregates.mean_faithfulness)}</span>
              </div>
              <div className="kpi-tile">
                <span className="kpi-label">Mean generation latency</span>
                <span className="kpi-value">
                  {(job.aggregates.mean_generation_latency_ms / 1000).toFixed(1)}s
                </span>
              </div>
              <div className="kpi-tile">
                <span className="kpi-label">Mean judge latency</span>
                <span className="kpi-value">
                  {(job.aggregates.mean_eval_latency_ms / 1000).toFixed(1)}s
                </span>
              </div>
              <div className="kpi-tile">
                <span className="kpi-label">Generation tokens (in/out)</span>
                <span className="kpi-value">
                  {job.aggregates.total_generation_input_tokens} /{" "}
                  {job.aggregates.total_generation_output_tokens}
                </span>
              </div>
              <div className="kpi-tile">
                <span className="kpi-label">Judge tokens (in/out)</span>
                <span className="kpi-value">
                  {job.aggregates.total_eval_input_tokens} /{" "}
                  {job.aggregates.total_eval_output_tokens}
                </span>
              </div>
            </div>
          )}

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
              {job.per_query.map((r) => (
                <EvalQueryRow
                  key={r.id}
                  r={r}
                  isOpen={expanded.has(r.id)}
                  onToggle={() => toggleExpanded(r.id)}
                />
              ))}
              {Array.from({ length: job.total - job.per_query.length }).map((_, i) => (
                <tr key={`pending-${i}`}>
                  <td colSpan={6}>pending...</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <hr />
      <ChatPanel />
    </div>
  );
}

export function EvalQueryRow({
  r,
  isOpen,
  onToggle,
}: {
  r: EvalQueryResult;
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
        <td>{r.error ? <span className="error">{r.error}</span> : r.hit ? "yes" : "no"}</td>
        <td>{r.error ? "-" : formatPercent(r.correctness)}</td>
        <td>{r.error ? "-" : formatPercent(r.faithfulness)}</td>
        <td>{(r.generation_latency_ms / 1000).toFixed(1)}s</td>
      </tr>
      {isOpen && (
        <tr>
          <td colSpan={6}>
            <div className="eval-detail">
              <div className="detail-item">
                <div className="detail-label">MRR</div>
                <div className="detail-value">{r.error ? "-" : r.mrr.toFixed(2)}</div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Citation accuracy</div>
                <div className="detail-value">
                  {r.error ? "-" : formatPercent(r.citation_accuracy)}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Citation recall</div>
                <div className="detail-value">
                  {r.error ? "-" : formatPercent(r.citation_recall)}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Answer relevance</div>
                <div className="detail-value">
                  {r.error ? "-" : formatPercent(r.answer_relevance)}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Info recall</div>
                <div className="detail-value">{r.error ? "-" : formatPercent(r.info_recall)}</div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Citation completeness</div>
                <div className="detail-value">
                  {r.error ? "-" : formatPercent(r.citation_completeness)}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Judge latency</div>
                <div className="detail-value">
                  {r.error ? "-" : `${(r.eval_latency_ms / 1000).toFixed(1)}s`}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Generation tokens (in/out)</div>
                <div className="detail-value">
                  {r.error
                    ? "-"
                    : `${r.generation_input_tokens ?? "-"} / ${r.generation_output_tokens ?? "-"}`}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Judge tokens (in/out)</div>
                <div className="detail-value">
                  {r.error ? "-" : `${r.eval_input_tokens ?? "-"} / ${r.eval_output_tokens ?? "-"}`}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-label">LLM calls</div>
                <div className="detail-value">{r.error ? "-" : r.llm_call_count}</div>
              </div>
              <div className="detail-item detail-answer">
                <div className="detail-label">Generated answer</div>
                <div className="detail-value">{r.error || r.generated_answer || "-"}</div>
              </div>
              {r.faithfulness_reasoning && (
                <div className="detail-item detail-answer">
                  <div className="detail-label">Faithfulness reasoning</div>
                  <div className="detail-value">{r.faithfulness_reasoning}</div>
                </div>
              )}
              {r.answer_relevance_reasoning && (
                <div className="detail-item detail-answer">
                  <div className="detail-label">Answer relevance reasoning</div>
                  <div className="detail-value">{r.answer_relevance_reasoning}</div>
                </div>
              )}
              {r.info_recall_reasoning && (
                <div className="detail-item detail-answer">
                  <div className="detail-label">Info recall reasoning</div>
                  <div className="detail-value">{r.info_recall_reasoning}</div>
                </div>
              )}
              {r.citation_completeness_reasoning && (
                <div className="detail-item detail-answer">
                  <div className="detail-label">Citation completeness reasoning</div>
                  <div className="detail-value">{r.citation_completeness_reasoning}</div>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </Fragment>
  );
}

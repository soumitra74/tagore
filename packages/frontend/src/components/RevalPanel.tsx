import { useCallback, useEffect, useRef, useState } from "react";
import { getEvalJudge, getEvalRun, listEvalHistory, startReval } from "../api/client";
import type { EvalCycle, EvalJobStatus, EvalJudgeInfo } from "../api/client";
import { EvalQueryRow } from "./EvalPanel";

const POLL_INTERVAL_MS = 2000;

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

function formatTimestamp(isoString: string): string {
  return new Date(isoString).toLocaleString();
}

function cycleLabel(cycle: EvalCycle): string {
  const { config } = cycle;
  return (
    `Run #${cycle.id} — ${config.chunking_strategy} ` +
    `(${config.chunk_size}/${config.chunk_overlap}), ` +
    `${config.embedding_provider}: ${config.embedding_model} — ` +
    `${cycle.total} questions — ${formatTimestamp(cycle.completed_at ?? cycle.started_at)}`
  );
}

export function RevalPanel() {
  const [cycles, setCycles] = useState<EvalCycle[]>([]);
  const [selectedCycleId, setSelectedCycleId] = useState<number | null>(null);
  const [judge, setJudge] = useState<EvalJudgeInfo | null>(null);
  const [job, setJob] = useState<EvalJobStatus | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    listEvalHistory()
      .then((history) => {
        const completed = history.filter((c) => c.status === "completed" && c.aggregates != null);
        setCycles(completed);
        if (completed.length > 0) setSelectedCycleId(completed[0].id);
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

  async function handleReval() {
    if (selectedCycleId === null) return;
    stopPolling();
    setStarting(true);
    setError(null);
    setJob(null);
    setExpanded(new Set());
    try {
      const { job_id } = await startReval(selectedCycleId);
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
    <div className="reval-panel">
      <h2>Reval</h2>
      <p>
        Re-judge a previous eval run's existing questions and generated answers with the
        currently active judge -- no new retrieval or generation happens, only fresh judge
        scores. Produces a new eval cycle so the original stays intact for comparison.
      </p>
      {judge && (
        <p>
          Judge LLM: {judge.provider}: {judge.model} (
          {judge.combine_calls ? "combined judge call" : "separate judge calls"})
        </p>
      )}

      {cycles.length === 0 && !error && <p>No fully completed eval runs to reval yet.</p>}

      {cycles.length > 0 && (
        <form
          className="config-form"
          onSubmit={(e) => {
            e.preventDefault();
            handleReval();
          }}
        >
          <label>
            Run to reval
            <select
              value={selectedCycleId ?? ""}
              onChange={(e) => setSelectedCycleId(Number(e.target.value))}
            >
              {cycles.map((cycle) => (
                <option key={cycle.id} value={cycle.id}>
                  {cycleLabel(cycle)}
                </option>
              ))}
            </select>
          </label>
          <button type="submit" disabled={starting || selectedCycleId === null}>
            {starting ? "Starting..." : "Start Reval"}
          </button>
        </form>
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
                <span className="kpi-label">Mean judge latency</span>
                <span className="kpi-value">
                  {(job.aggregates.mean_eval_latency_ms / 1000).toFixed(1)}s
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
    </div>
  );
}

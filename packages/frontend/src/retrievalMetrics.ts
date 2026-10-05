// retrieval_metrics is a free-form map keyed "<metric>@<k>" (e.g. "recall@5",
// "ndcg@10") so a new metric is just another key on the backend -- these
// helpers are the only place the UI needs to know how to order, label and
// format one. An unrecognized metric still renders: it sorts after the known
// ones, keeps its raw name as the label, and falls back to percent format.

const METRIC_ORDER = ["recall", "ndcg", "rerank_recall", "rerank_ndcg"];
const METRIC_LABELS: Record<string, string> = {
  recall: "Recall",
  ndcg: "nDCG",
  rerank_recall: "Reranked recall",
  rerank_ndcg: "Reranked nDCG",
};

function parseKey(key: string): { metric: string; k: number } {
  const [metric, k] = key.split("@");
  return { metric, k: Number(k) };
}

function metricRank(metric: string): number {
  const index = METRIC_ORDER.indexOf(metric);
  return index === -1 ? METRIC_ORDER.length : index;
}

// Grouped by metric (recall before nDCG), then numerically by k -- a plain
// string sort would put "recall@10" before "recall@5".
export function sortRetrievalMetricKeys(keys: string[]): string[] {
  return [...keys].sort((a, b) => {
    const pa = parseKey(a);
    const pb = parseKey(b);
    return metricRank(pa.metric) - metricRank(pb.metric) || pa.k - pb.k;
  });
}

export function retrievalMetricLabel(key: string): string {
  const { metric, k } = parseKey(key);
  return `${METRIC_LABELS[metric] ?? metric}@${k}`;
}

// Every metric in the group is a 0-1 fraction, shown as a percent.
export function formatRetrievalMetric(_key: string, value: number | null | undefined): string {
  if (value == null) return "-";
  return `${(value * 100).toFixed(0)}%`;
}

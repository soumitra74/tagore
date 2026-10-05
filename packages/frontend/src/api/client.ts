const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:9099";

export interface Source {
  document: string;
  page: number;
  snippet: string;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
}

export interface IngestResponse {
  document_id: string;
  chunk_count: number;
  page_count: number;
  config_id: number;
  metrics: IngestionMetrics;
}

export interface DocumentRecord {
  id: string;
  filename: string;
  uploaded_at: string;
}

export interface ExperimentConfig {
  id: number;
  chunking_strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  embedding_provider: string;
  embedding_model: string;
  embedding_dim: number;
}

// embedding_dim is a property of the model, not an independent user choice
// -- the backend derives it server-side (app/providers/embedding_catalog.py)
// from embedding_provider + embedding_model, so it's not part of what the
// client sends when creating a config.
export type CreateConfigRequest = Omit<ExperimentConfig, "id" | "embedding_dim">;

// Known embedding models per provider and the fixed output dimension each
// one produces. Mirrors the backend's app/providers/embedding_catalog.py
// (itself mirroring app/providers/local_embedding_provider.py's
// LOCAL_EMBEDDING_MODELS for "local") -- kept here too so the model dropdown
// can show each model's dimension without a round trip.
export const EMBEDDING_MODEL_CATALOG: Record<string, { model: string; dim: number }[]> = {
  local: [
    { model: "BAAI/bge-base-en-v1.5", dim: 768 },
    { model: "intfloat/e5-base-v2", dim: 768 },
    { model: "nomic-ai/nomic-embed-text-v1.5", dim: 768 },
    { model: "BAAI/bge-m3", dim: 1024 },
  ],
  openai: [
    { model: "text-embedding-3-small", dim: 1536 },
    { model: "text-embedding-3-large", dim: 3072 },
    { model: "text-embedding-ada-002", dim: 1536 },
  ],
  ollama: [
    { model: "nomic-embed-text", dim: 768 },
    { model: "mxbai-embed-large", dim: 1024 },
    { model: "all-minilm", dim: 384 },
  ],
};

export function getExpectedEmbeddingDim(provider: string, model: string): number | undefined {
  return EMBEDDING_MODEL_CATALOG[provider]?.find((m) => m.model === model)?.dim;
}

export function isValidEmbeddingCombo(provider: string, model: string, dim: number): boolean {
  return getExpectedEmbeddingDim(provider, model) === dim;
}

export interface CorpusFileIngestResult {
  filename: string;
  document_id?: string;
  chunk_count?: number;
  page_count?: number;
  error?: string;
}

export interface IngestionMetrics {
  chunks: number;
  total_pages: number;
  total_time_ms: number;
  error_count: number;
  input_size_bytes: number;
  vector_db_size_bytes: number;
}

export interface IngestConfigResponse {
  config: ExperimentConfig;
  results: CorpusFileIngestResult[];
  metrics: IngestionMetrics;
}

export interface IngestionRun {
  id: number;
  ran_at: string;
  config: ExperimentConfig;
  results: CorpusFileIngestResult[];
  total_chunks: number;
  total_pages: number;
  total_time_ms: number;
  error_count: number;
  input_size_bytes: number;
  vector_db_size_bytes: number;
}

export interface EvalQueryResult {
  id: string;
  question: string;
  hit: boolean;
  mrr: number;
  citation_accuracy: number;
  citation_recall: number;
  correctness: number;
  correctness_reasoning: string;
  faithfulness: number;
  faithfulness_reasoning: string;
  answer_relevance: number;
  answer_relevance_reasoning: string;
  info_recall: number;
  info_recall_reasoning: string;
  citation_completeness: number;
  citation_completeness_reasoning: string;
  generated_answer: string;
  generation_latency_ms: number;
  eval_latency_ms: number;
  generation_input_tokens: number | null;
  generation_output_tokens: number | null;
  eval_input_tokens: number | null;
  eval_output_tokens: number | null;
  llm_call_count: number;
  error?: string | null;
}

export interface EvalAggregates {
  num_queries: number;
  hit_rate: number;
  mean_mrr: number;
  mean_citation_accuracy: number;
  mean_citation_recall: number;
  mean_retrieval_metrics: Record<string, number> | null;
  mean_rerank_latency_ms?: number | null;
  mean_correctness: number;
  mean_faithfulness: number;
  mean_answer_relevance: number;
  mean_info_recall: number;
  mean_citation_completeness: number;
  mean_generation_latency_ms: number;
  mean_eval_latency_ms: number;
  total_generation_input_tokens: number;
  total_generation_output_tokens: number;
  total_eval_input_tokens: number;
  total_eval_output_tokens: number;
  total_llm_calls: number;
}

export interface EvalJobStatus {
  job_id: string;
  config_id: number;
  total: number;
  status: "running" | "completed" | "failed" | "aborted";
  per_query: EvalQueryResult[];
  aggregates: EvalAggregates | null;
  error?: string | null;
}

export interface EvalCycleQueryResult {
  id: number;
  question: string;
  generated_answer: string;
  hit_rate: number;
  mrr: number;
  citation_accuracy: number;
  citation_recall: number | null;
  retrieval_metrics: Record<string, number> | null;
  rerank_latency_ms?: number | null;
  correctness: number;
  faithfulness: number | null;
  answer_relevance: number | null;
  info_recall: number | null;
  citation_completeness: number | null;
  generation_latency_ms: number;
  eval_latency_ms: number;
  generation_input_tokens: number | null;
  generation_output_tokens: number | null;
  eval_input_tokens: number | null;
  eval_output_tokens: number | null;
  llm_call_count: number | null;
  run_at: string;
}

export interface EvalCycle {
  id: number;
  status: "running" | "completed" | "failed" | "aborted";
  total: number;
  aggregates: EvalAggregates | null;
  started_at: string;
  completed_at: string | null;
  judge_provider: string | null;
  judge_model: string | null;
  graph_augmented: boolean;
  rerank_model?: string | null;
  rerank_candidates?: number | null;
  config: ExperimentConfig;
  per_query: EvalCycleQueryResult[];
}

export interface EvalJudgeInfo {
  provider: string;
  model: string;
  combine_calls: boolean;
}

async function parseErrorDetail(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    return body.detail ?? fallback;
  } catch {
    return fallback;
  }
}

export async function listDocuments(): Promise<DocumentRecord[]> {
  const res = await fetch(`${API_BASE_URL}/api/documents`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to list documents (${res.status})`));
  }
  return res.json();
}

export async function askQuestion(question: string, configId?: number): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, config_id: configId ?? null }),
  });
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Chat failed (${res.status})`));
  }
  return res.json();
}

export async function listConfigs(): Promise<ExperimentConfig[]> {
  const res = await fetch(`${API_BASE_URL}/api/configs`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to list configs (${res.status})`));
  }
  return res.json();
}

export async function ingestConfigCorpus(
  payload: CreateConfigRequest,
): Promise<IngestConfigResponse> {
  const res = await fetch(`${API_BASE_URL}/api/corpus/ingest-config`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Ingestion failed (${res.status})`));
  }
  return res.json();
}

export async function listCorpusFiles(): Promise<string[]> {
  const res = await fetch(`${API_BASE_URL}/api/corpus`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to list corpus files (${res.status})`));
  }
  return res.json();
}

export async function listIngestionRuns(): Promise<IngestionRun[]> {
  const res = await fetch(`${API_BASE_URL}/api/ingestion-runs`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to list ingestion runs (${res.status})`));
  }
  return res.json();
}

export async function getEvalJudge(): Promise<EvalJudgeInfo> {
  const res = await fetch(`${API_BASE_URL}/api/eval/judge`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to get judge info (${res.status})`));
  }
  return res.json();
}

// Mirrors the backend's RERANKER_MODELS (app/providers/reranker.py).
export const RERANK_MODELS = ["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base"];
export const DEFAULT_RERANK_CANDIDATES = 20;

export async function startEvalRun(
  configId: number,
  topK?: number,
  graphAugmented?: boolean,
  rerank?: { model: string; candidates: number },
): Promise<{ job_id: string; config_id: number; total: number }> {
  const res = await fetch(`${API_BASE_URL}/api/eval/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      config_id: configId,
      top_k: topK ?? null,
      graph_augmented: graphAugmented ?? false,
      rerank_model: rerank?.model ?? null,
      rerank_candidates: rerank?.candidates ?? null,
    }),
  });
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to start eval run (${res.status})`));
  }
  return res.json();
}

export async function startReval(
  sourceCycleId: number,
): Promise<{ job_id: string; config_id: number; total: number; cycle_id: number }> {
  const res = await fetch(`${API_BASE_URL}/api/eval/reval`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_cycle_id: sourceCycleId }),
  });
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to start reval (${res.status})`));
  }
  return res.json();
}

export async function getEvalRun(jobId: string): Promise<EvalJobStatus> {
  const res = await fetch(`${API_BASE_URL}/api/eval/run/${jobId}`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to get eval run (${res.status})`));
  }
  return res.json();
}

export async function listEvalHistory(): Promise<EvalCycle[]> {
  const res = await fetch(`${API_BASE_URL}/api/eval/history`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to list eval history (${res.status})`));
  }
  return res.json();
}

export async function deleteEvalCycle(cycleId: number): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/eval/history/${cycleId}`, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to hide eval cycle (${res.status})`));
  }
}

export async function purgeEvalCycle(cycleId: number): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/eval/history/${cycleId}/purge`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to purge eval cycle (${res.status})`));
  }
}

export interface EvalQuery {
  id: number;
  query_text: string;
  query_type: string | null;
}

export async function listEvalQueries(): Promise<EvalQuery[]> {
  const res = await fetch(`${API_BASE_URL}/api/eval/queries`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to list eval queries (${res.status})`));
  }
  return res.json();
}

export interface RcaToolCall {
  tool: string;
  args: Record<string, unknown>;
  result_summary: string;
}

export interface RcaMessage {
  role: "user" | "assistant";
  content: string;
  tool_trace: RcaToolCall[] | null;
}

export interface RcaSession {
  id: number;
  cycle_id: number;
  query_id: number;
  total_input_tokens: number;
  total_output_tokens: number;
}

export async function listRcaSessions(): Promise<RcaSession[]> {
  const res = await fetch(`${API_BASE_URL}/api/rca/sessions`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to list RCA sessions (${res.status})`));
  }
  return res.json();
}

export async function deleteRcaSession(sessionId: number): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/rca/sessions/${sessionId}`, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to delete RCA session (${res.status})`));
  }
}

export async function createRcaSession(
  cycleId: number,
  queryId: number,
): Promise<{ session_id: number; message: RcaMessage }> {
  const res = await fetch(`${API_BASE_URL}/api/rca/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cycle_id: cycleId, query_id: queryId }),
  });
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to start RCA session (${res.status})`));
  }
  return res.json();
}

export async function listRcaMessages(sessionId: number): Promise<RcaMessage[]> {
  const res = await fetch(`${API_BASE_URL}/api/rca/sessions/${sessionId}/messages`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to load RCA session (${res.status})`));
  }
  return res.json();
}

export async function sendRcaMessage(sessionId: number, message: string): Promise<RcaMessage> {
  const res = await fetch(`${API_BASE_URL}/api/rca/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to send RCA message (${res.status})`));
  }
  return res.json();
}

export interface ChunkDocument {
  document_id: string;
  filename: string;
}

export interface StoredChunk {
  id: string;
  page_number: number;
  chunk_text: string;
  context_text: string | null;
  filename: string;
}

export interface ChunksPage {
  chunks: StoredChunk[];
  total: number;
  page: number;
  page_size: number;
}

export async function listChunkDocuments(configId: number): Promise<ChunkDocument[]> {
  const res = await fetch(`${API_BASE_URL}/api/chunks/documents?config_id=${configId}`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to list documents (${res.status})`));
  }
  return res.json();
}

export interface ChunkLookupResult {
  chunks: StoredChunk[];
  not_found: string[];
}

export async function lookupChunks(configId: number, ids: string): Promise<ChunkLookupResult> {
  const query = new URLSearchParams({ config_id: String(configId), ids });
  const res = await fetch(`${API_BASE_URL}/api/chunks/lookup?${query.toString()}`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to look up chunks (${res.status})`));
  }
  return res.json();
}

export async function listChunks(params: {
  configId: number;
  documentId?: string;
  search?: string;
  page: number;
  pageSize: number;
}): Promise<ChunksPage> {
  const query = new URLSearchParams({
    config_id: String(params.configId),
    page: String(params.page),
    page_size: String(params.pageSize),
  });
  if (params.documentId) query.set("document_id", params.documentId);
  if (params.search) query.set("search", params.search);

  const res = await fetch(`${API_BASE_URL}/api/chunks?${query.toString()}`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to list chunks (${res.status})`));
  }
  return res.json();
}

export async function ingestCorpusFile(
  filename: string,
  configId?: number,
): Promise<IngestResponse> {
  const res = await fetch(`${API_BASE_URL}/api/corpus/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, config_id: configId ?? null }),
  });
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Ingestion failed (${res.status})`));
  }
  return res.json();
}

export interface GraphEntity {
  entity_type: string;
  entity_text: string;
  chunk_id: string;
  page_number: number;
  filename: string;
}

export interface GraphEntitiesPage {
  entities: GraphEntity[];
  total: number;
  page: number;
  page_size: number;
}

export interface GraphCooccurrence {
  entity_type_a: string;
  entity_text_a: string;
  entity_type_b: string;
  entity_text_b: string;
}

export interface GraphCooccurrencesPage {
  cooccurrences: GraphCooccurrence[];
  total: number;
  page: number;
  page_size: number;
}

export async function listGraphEntities(params: {
  configId: number;
  documentId: string;
  page: number;
  pageSize: number;
}): Promise<GraphEntitiesPage> {
  const query = new URLSearchParams({
    config_id: String(params.configId),
    document_id: params.documentId,
    page: String(params.page),
    page_size: String(params.pageSize),
  });
  const res = await fetch(`${API_BASE_URL}/api/graph/entities?${query.toString()}`);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to list graph entities (${res.status})`));
  }
  return res.json();
}

export async function listGraphCooccurrences(params: {
  documentId: string;
  entityType?: string;
  page: number;
  pageSize: number;
}): Promise<GraphCooccurrencesPage> {
  const query = new URLSearchParams({
    document_id: params.documentId,
    page: String(params.page),
    page_size: String(params.pageSize),
  });
  if (params.entityType) query.set("entity_type", params.entityType);
  const res = await fetch(`${API_BASE_URL}/api/graph/cooccurrences?${query.toString()}`);
  if (!res.ok) {
    throw new Error(
      await parseErrorDetail(res, `Failed to list graph cooccurrences (${res.status})`),
    );
  }
  return res.json();
}

export async function backfillGraph(
  configId: number,
): Promise<{
  chunks_processed: number;
  entities_touched: number;
  mentions_created: number;
  cooccurrences_created: number;
}> {
  const res = await fetch(`${API_BASE_URL}/api/graph/backfill`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config_id: configId }),
  });
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res, `Failed to backfill graph (${res.status})`));
  }
  return res.json();
}

import { Fragment, useEffect, useState } from "react";
import { listChunkDocuments, listChunks, listConfigs, lookupChunks } from "../api/client";
import type {
  ChunkDocument,
  ChunkLookupResult,
  ChunksPage,
  ExperimentConfig,
  StoredChunk,
} from "../api/client";

const PAGE_SIZE = 25;
const PREVIEW_LENGTH = 160;

function truncate(text: string): string {
  return text.length > PREVIEW_LENGTH ? `${text.slice(0, PREVIEW_LENGTH)}...` : text;
}

export function ChunksPanel() {
  const [configs, setConfigs] = useState<ExperimentConfig[]>([]);
  const [selectedConfigId, setSelectedConfigId] = useState<number | null>(null);
  const [documents, setDocuments] = useState<ChunkDocument[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string>("");
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [result, setResult] = useState<ChunksPage | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [lookupInput, setLookupInput] = useState("");
  const [lookupResult, setLookupResult] = useState<ChunkLookupResult | null>(null);
  const [lookupExpanded, setLookupExpanded] = useState<Set<string>>(new Set());
  const [lookupError, setLookupError] = useState<string | null>(null);
  const [lookingUp, setLookingUp] = useState(false);

  useEffect(() => {
    listConfigs()
      .then((result) => {
        setConfigs(result);
        if (result.length > 0) setSelectedConfigId(result[0].id);
      })
      .catch((err) => setError((err as Error).message));
  }, []);

  useEffect(() => {
    if (selectedConfigId === null) return;
    listChunkDocuments(selectedConfigId)
      .then(setDocuments)
      .catch((err) => setError((err as Error).message));
  }, [selectedConfigId]);

  function handleConfigChange(configId: number) {
    setSelectedConfigId(configId);
    setSelectedDocumentId("");
    setSearchInput("");
    setSearch("");
    setPage(1);
  }

  useEffect(() => {
    if (selectedConfigId === null) return;
    setLoading(true);
    setError(null);
    listChunks({
      configId: selectedConfigId,
      documentId: selectedDocumentId || undefined,
      search: search || undefined,
      page,
      pageSize: PAGE_SIZE,
    })
      .then((page) => {
        setResult(page);
        setExpanded(new Set());
      })
      .catch((err) => setError((err as Error).message))
      .finally(() => setLoading(false));
  }, [selectedConfigId, selectedDocumentId, search, page]);

  function toggleExpanded(id: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function handleSearchSubmit(e: React.FormEvent) {
    e.preventDefault();
    setPage(1);
    setSearch(searchInput);
  }

  function toggleLookupExpanded(id: string) {
    setLookupExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function handleLookupSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (selectedConfigId === null || !lookupInput.trim()) return;
    setLookingUp(true);
    setLookupError(null);
    try {
      const result = await lookupChunks(selectedConfigId, lookupInput);
      setLookupResult(result);
      setLookupExpanded(new Set());
    } catch (err) {
      setLookupError((err as Error).message);
    } finally {
      setLookingUp(false);
    }
  }

  const totalPages = result ? Math.max(1, Math.ceil(result.total / result.page_size)) : 1;

  return (
    <div className="chunks-panel">
      <h2>Chunks</h2>
      <p>Inspect the chunks a config actually produced -- for debugging chunking behavior.</p>

      <form className="config-form" onSubmit={handleSearchSubmit}>
        <label>
          Config
          <select
            value={selectedConfigId ?? ""}
            onChange={(e) => handleConfigChange(Number(e.target.value))}
          >
            {configs.map((c) => (
              <option key={c.id} value={c.id}>
                {c.id} ({c.chunking_strategy}, {c.chunk_size}/{c.chunk_overlap},{" "}
                {c.embedding_provider}: {c.embedding_model})
              </option>
            ))}
          </select>
        </label>
        <label>
          Document
          <select
            value={selectedDocumentId}
            onChange={(e) => {
              setSelectedDocumentId(e.target.value);
              setPage(1);
            }}
          >
            <option value="">All documents</option>
            {documents.map((d) => (
              <option key={d.document_id} value={d.document_id}>
                {d.filename}
              </option>
            ))}
          </select>
        </label>
        <label>
          Search chunk text
          <input
            type="text"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            placeholder="e.g. rotor blade"
          />
        </label>
        <button type="submit">Search</button>
      </form>

      {error && <p className="error">{error}</p>}
      {loading && <p>Loading...</p>}

      {result && (
        <>
          <p>
            {result.total} chunk{result.total === 1 ? "" : "s"} -- page {result.page} of{" "}
            {totalPages}
          </p>
          <table className="corpus-table">
            <thead>
              <tr>
                <th></th>
                <th>Page</th>
                <th>Filename</th>
                <th>Chunk text</th>
              </tr>
            </thead>
            <tbody>
              {result.chunks.map((chunk) => (
                <ChunkRow
                  key={chunk.id}
                  chunk={chunk}
                  isOpen={expanded.has(chunk.id)}
                  onToggle={() => toggleExpanded(chunk.id)}
                />
              ))}
              {result.chunks.length === 0 && (
                <tr>
                  <td colSpan={4}>No chunks match these filters.</td>
                </tr>
              )}
            </tbody>
          </table>
          <div className="pagination">
            <button type="button" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
              Previous
            </button>
            <span>
              Page {page} of {totalPages}
            </span>
            <button
              type="button"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </button>
          </div>
        </>
      )}

      <hr />
      <h3>Look up chunks by ID</h3>
      <p>
        Cross-check a specific chunk id from elsewhere (an eval run's retrieved chunks, a
        citation) against what's actually stored -- bypasses the filters/pagination above.
      </p>
      <form className="config-form" onSubmit={handleLookupSubmit}>
        <label>
          Chunk ID(s), comma-separated
          <input
            type="text"
            value={lookupInput}
            onChange={(e) => setLookupInput(e.target.value)}
            placeholder="e.g. 11111111-1111-1111-1111-111111111111, 22222222-..."
          />
        </label>
        <button type="submit" disabled={lookingUp || selectedConfigId === null}>
          {lookingUp ? "Looking up..." : "Look up"}
        </button>
      </form>
      {lookupError && <p className="error">{lookupError}</p>}
      {lookupResult && (
        <>
          {lookupResult.not_found.length > 0 && (
            <p className="error">Not found: {lookupResult.not_found.join(", ")}</p>
          )}
          {lookupResult.chunks.length > 0 && (
            <table className="corpus-table">
              <thead>
                <tr>
                  <th></th>
                  <th>Page</th>
                  <th>Filename</th>
                  <th>Chunk text</th>
                </tr>
              </thead>
              <tbody>
                {lookupResult.chunks.map((chunk) => (
                  <ChunkRow
                    key={chunk.id}
                    chunk={chunk}
                    isOpen={lookupExpanded.has(chunk.id)}
                    onToggle={() => toggleLookupExpanded(chunk.id)}
                  />
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}

function ChunkRow({
  chunk,
  isOpen,
  onToggle,
}: {
  chunk: StoredChunk;
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
        <td>{chunk.page_number}</td>
        <td>{chunk.filename}</td>
        <td>{truncate(chunk.chunk_text)}</td>
      </tr>
      {isOpen && (
        <tr>
          <td colSpan={4}>
            <div className="eval-detail">
              <div className="detail-item detail-answer">
                <div className="detail-label">Chunk id</div>
                <div className="detail-value">{chunk.id}</div>
              </div>
              <div className="detail-item detail-answer">
                <div className="detail-label">Chunk text (embedded)</div>
                <div className="detail-value">{chunk.chunk_text}</div>
              </div>
              {chunk.context_text && (
                <div className="detail-item detail-answer">
                  <div className="detail-label">
                    Context text (shown to generator, not embedded)
                  </div>
                  <div className="detail-value">{chunk.context_text}</div>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </Fragment>
  );
}

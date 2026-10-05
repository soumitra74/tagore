import { useEffect, useState } from "react";
import {
  backfillGraph,
  listChunkDocuments,
  listConfigs,
  listGraphCooccurrences,
  listGraphEntities,
} from "../api/client";
import type {
  ChunkDocument,
  ExperimentConfig,
  GraphCooccurrencesPage,
  GraphEntitiesPage,
} from "../api/client";

const PAGE_SIZE = 25;

const ENTITY_TYPES = ["REPORT_NUMBER", "DATA_COUNTER_ID", "REPORT_DATE", "AIRCRAFT_TYPE"];

export function GraphPanel() {
  const [configs, setConfigs] = useState<ExperimentConfig[]>([]);
  const [selectedConfigId, setSelectedConfigId] = useState<number | null>(null);
  const [documents, setDocuments] = useState<ChunkDocument[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string>("");
  const [entitiesPage, setEntitiesPage] = useState(1);
  const [cooccurrencesPage, setCooccurrencesPage] = useState(1);
  const [cooccurrenceEntityType, setCooccurrenceEntityType] = useState<string>("");
  const [entities, setEntities] = useState<GraphEntitiesPage | null>(null);
  const [cooccurrences, setCooccurrences] = useState<GraphCooccurrencesPage | null>(null);
  const [backfilling, setBackfilling] = useState(false);
  const [backfillResult, setBackfillResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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
      .then((docs) => {
        setDocuments(docs);
        if (docs.length > 0) setSelectedDocumentId(docs[0].document_id);
      })
      .catch((err) => setError((err as Error).message));
  }, [selectedConfigId]);

  function handleConfigChange(configId: number) {
    setSelectedConfigId(configId);
    setSelectedDocumentId("");
    setEntitiesPage(1);
    setCooccurrencesPage(1);
  }

  useEffect(() => {
    if (selectedConfigId === null || !selectedDocumentId) return;
    listGraphEntities({
      configId: selectedConfigId,
      documentId: selectedDocumentId,
      page: entitiesPage,
      pageSize: PAGE_SIZE,
    })
      .then(setEntities)
      .catch((err) => setError((err as Error).message));
  }, [selectedConfigId, selectedDocumentId, entitiesPage]);

  useEffect(() => {
    if (!selectedDocumentId) return;
    listGraphCooccurrences({
      documentId: selectedDocumentId,
      entityType: cooccurrenceEntityType || undefined,
      page: cooccurrencesPage,
      pageSize: PAGE_SIZE,
    })
      .then(setCooccurrences)
      .catch((err) => setError((err as Error).message));
  }, [selectedDocumentId, cooccurrenceEntityType, cooccurrencesPage]);

  async function handleBackfill() {
    if (selectedConfigId === null) return;
    setBackfilling(true);
    setBackfillResult(null);
    setError(null);
    try {
      const result = await backfillGraph(selectedConfigId);
      setBackfillResult(
        `Processed ${result.chunks_processed} chunks -- ${result.entities_touched} entities, ` +
          `${result.mentions_created} mentions, ${result.cooccurrences_created} co-occurrences.`,
      );
      setEntitiesPage(1);
      setCooccurrencesPage(1);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBackfilling(false);
    }
  }

  const entitiesTotalPages = entities
    ? Math.max(1, Math.ceil(entities.total / entities.page_size))
    : 1;
  const cooccurrencesTotalPages = cooccurrences
    ? Math.max(1, Math.ceil(cooccurrences.total / cooccurrences.page_size))
    : 1;

  return (
    <div className="graph-panel">
      <h2>Graph</h2>
      <p>Inspect the entities and relationships extracted for a config's chunks.</p>

      <form className="config-form" onSubmit={(e) => e.preventDefault()}>
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
              setEntitiesPage(1);
              setCooccurrencesPage(1);
            }}
          >
            {documents.map((d) => (
              <option key={d.document_id} value={d.document_id}>
                {d.filename}
              </option>
            ))}
          </select>
        </label>
        <button type="button" onClick={handleBackfill} disabled={backfilling}>
          {backfilling ? "Backfilling..." : "Run entity extraction for this config"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}
      {backfillResult && <p>{backfillResult}</p>}

      <h3>Entities</h3>
      {entities && (
        <>
          <p>{entities.total} mention(s)</p>
          <table className="corpus-table">
            <thead>
              <tr>
                <th>Type</th>
                <th>Text</th>
                <th>Page</th>
                <th>Filename</th>
              </tr>
            </thead>
            <tbody>
              {entities.entities.map((e, i) => (
                <tr key={`${e.chunk_id}-${e.entity_type}-${e.entity_text}-${i}`}>
                  <td>{e.entity_type}</td>
                  <td>{e.entity_text}</td>
                  <td>{e.page_number}</td>
                  <td>{e.filename}</td>
                </tr>
              ))}
              {entities.entities.length === 0 && (
                <tr>
                  <td colSpan={4}>No entities extracted for this document yet.</td>
                </tr>
              )}
            </tbody>
          </table>
          <div className="pagination">
            <button
              type="button"
              disabled={entitiesPage <= 1}
              onClick={() => setEntitiesPage((p) => p - 1)}
            >
              Previous
            </button>
            <span>
              Page {entitiesPage} of {entitiesTotalPages}
            </span>
            <button
              type="button"
              disabled={entitiesPage >= entitiesTotalPages}
              onClick={() => setEntitiesPage((p) => p + 1)}
            >
              Next
            </button>
          </div>
        </>
      )}

      <h3>Co-occurrences</h3>
      <label>
        Filter by entity type
        <select
          value={cooccurrenceEntityType}
          onChange={(e) => {
            setCooccurrenceEntityType(e.target.value);
            setCooccurrencesPage(1);
          }}
        >
          <option value="">All types</option>
          {ENTITY_TYPES.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </select>
      </label>
      {cooccurrences && (
        <>
          <p>{cooccurrences.total} pair(s)</p>
          <table className="corpus-table">
            <thead>
              <tr>
                <th>Entity A</th>
                <th>Entity B</th>
              </tr>
            </thead>
            <tbody>
              {cooccurrences.cooccurrences.map((c, i) => (
                <tr key={i}>
                  <td>
                    {c.entity_type_a}: {c.entity_text_a}
                  </td>
                  <td>
                    {c.entity_type_b}: {c.entity_text_b}
                  </td>
                </tr>
              ))}
              {cooccurrences.cooccurrences.length === 0 && (
                <tr>
                  <td colSpan={2}>No co-occurrences for this document yet.</td>
                </tr>
              )}
            </tbody>
          </table>
          <div className="pagination">
            <button
              type="button"
              disabled={cooccurrencesPage <= 1}
              onClick={() => setCooccurrencesPage((p) => p - 1)}
            >
              Previous
            </button>
            <span>
              Page {cooccurrencesPage} of {cooccurrencesTotalPages}
            </span>
            <button
              type="button"
              disabled={cooccurrencesPage >= cooccurrencesTotalPages}
              onClick={() => setCooccurrencesPage((p) => p + 1)}
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
}

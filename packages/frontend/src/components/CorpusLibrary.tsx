import { useCallback, useEffect, useState } from "react";
import { listDocuments } from "../api/client";
import type { DocumentRecord } from "../api/client";

function formatTimestamp(isoString: string): string {
  return new Date(isoString).toLocaleString();
}

export function CorpusLibrary() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setDocuments(await listDocuments());
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="corpus-library">
      <h2>Ingested documents</h2>
      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}
      {!loading && !error && documents.length === 0 && <p>No documents ingested yet.</p>}
      {!loading && !error && documents.length > 0 && (
        <table className="corpus-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Uploaded</th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr key={doc.id}>
                <td>{doc.filename}</td>
                <td>{formatTimestamp(doc.uploaded_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

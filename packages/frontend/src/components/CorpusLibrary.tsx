import { useCallback, useEffect, useState } from "react";
import { listCorpusFiles, listDocuments } from "../api/client";
import type { DocumentRecord } from "../api/client";

function formatTimestamp(isoString: string): string {
  return new Date(isoString).toLocaleString();
}

export function CorpusLibrary() {
  const [files, setFiles] = useState<string[]>([]);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async (isCancelled?: () => boolean) => {
    setLoading(true);
    setError(null);
    const maxAttempts = 8;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const [corpusFiles, documents] = await Promise.all([
          listCorpusFiles(),
          listDocuments(),
        ]);
        if (isCancelled?.()) return;
        setFiles(corpusFiles);
        setDocuments(documents);
        setLoading(false);
        return;
      } catch (err) {
        if (isCancelled?.()) return;
        const message = (err as Error).message;
        const retryable = message === "Failed to fetch" && attempt < maxAttempts - 1;
        if (!retryable) {
          setError(message);
          setLoading(false);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
        if (isCancelled?.()) return;
      }
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    refresh(() => cancelled);
    return () => {
      cancelled = true;
    };
  }, [refresh]);

  return (
    <div className="corpus-library">
      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}
      <h2>Corpus files</h2>
      {!loading && !error && files.length === 0 && <p>No files found in the corpus/ folder.</p>}
      {!loading && !error && files.length > 0 && (
        <table className="corpus-table">
          <thead>
            <tr>
              <th>Filename</th>
            </tr>
          </thead>
          <tbody>
            {files.map((filename) => (
              <tr key={filename}>
                <td>{filename}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h2>Ingested documents</h2>
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

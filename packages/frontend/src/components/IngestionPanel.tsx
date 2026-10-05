import { useCallback, useEffect, useState } from "react";
import { listConfigs } from "../api/client";
import type { ExperimentConfig } from "../api/client";
import { ConfigsPanel } from "./ConfigsPanel";
import { CorpusFilesPanel } from "./CorpusFilesPanel";

export function IngestionPanel() {
  const [configs, setConfigs] = useState<ExperimentConfig[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      setConfigs(await listConfigs());
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="ingestion-panel">
      {error && <p className="error">{error}</p>}

      <ConfigsPanel configs={configs} onIngested={refresh} />
      <CorpusFilesPanel configs={configs} onIngested={refresh} />
    </div>
  );
}

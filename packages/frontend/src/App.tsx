import { useEffect, useState } from "react";
import "./App.css";
import { ChunksPanel } from "./components/ChunksPanel";
import { CorpusLibrary } from "./components/CorpusLibrary";
import { EvalComparePanel } from "./components/EvalComparePanel";
import { EvalHistoryPanel } from "./components/EvalHistoryPanel";
import { EvalPanel } from "./components/EvalPanel";
import { GraphPanel } from "./components/GraphPanel";
import { IngestionHistoryPanel } from "./components/IngestionHistoryPanel";
import { IngestionPanel } from "./components/IngestionPanel";
import { RCAPanel } from "./components/RCAPanel";
import { RevalPanel } from "./components/RevalPanel";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:9099";

const TABS = [
  { id: "corpus", label: "Corpus Library" },
  { id: "ingestion", label: "Ingestion" },
  { id: "history", label: "Ingestion History" },
  { id: "chunks", label: "Chunks" },
  { id: "graph", label: "Graph" },
  { id: "eval", label: "Eval" },
  { id: "reval", label: "Reval" },
  { id: "eval-history", label: "Eval History" },
  { id: "eval-compare", label: "Compare Evals" },
  { id: "rca", label: "RCA" },
] as const;

type TabId = (typeof TABS)[number]["id"];

const ACTIVE_TAB_STORAGE_KEY = "activeTab";

function isTabId(value: unknown): value is TabId {
  return TABS.some((tab) => tab.id === value);
}

function getStoredTab(): TabId {
  try {
    const stored = localStorage.getItem(ACTIVE_TAB_STORAGE_KEY);
    if (isTabId(stored)) return stored;
  } catch {
    // localStorage unavailable (private browsing, blocked site data, etc.) -- fall through.
  }
  return "corpus";
}

function App() {
  const [backendStatus, setBackendStatus] = useState("checking...");
  const [activeTab, setActiveTab] = useState<TabId>(getStoredTab);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/health`)
      .then((res) => res.json())
      .then((data) => setBackendStatus(data.status))
      .catch(() => setBackendStatus("unreachable"));
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(ACTIVE_TAB_STORAGE_KEY, activeTab);
    } catch {
      // Ignore -- persisting the tab is a convenience, not a requirement.
    }
  }, [activeTab]);

  return (
    <div className="app">
      <h1>PDF RAG</h1>
      <p>Backend status: {backendStatus}</p>
      <nav className="tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={tab.id === activeTab ? "tab active" : "tab"}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      <div className="tab-content">
        {activeTab === "corpus" && <CorpusLibrary />}
        {activeTab === "ingestion" && <IngestionPanel />}
        {activeTab === "history" && <IngestionHistoryPanel />}
        {activeTab === "chunks" && <ChunksPanel />}
        {activeTab === "graph" && <GraphPanel />}
        {activeTab === "eval" && <EvalPanel />}
        {activeTab === "reval" && <RevalPanel />}
        {activeTab === "eval-history" && <EvalHistoryPanel />}
        {activeTab === "eval-compare" && <EvalComparePanel />}
        {activeTab === "rca" && <RCAPanel />}
      </div>
    </div>
  );
}

export default App;

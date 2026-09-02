import { useState, useRef } from "react";
import Papa from "papaparse";
import { Upload, Play } from "lucide-react";
import ResultsTable from "../components/predictions/ResultsTable.jsx";
import { predictBatch } from "../services/api.js";

export default function TrafficAnalysis() {
  const [rows, setRows] = useState([]);
  const [fileName, setFileName] = useState(null);
  const [results, setResults] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    setFileName(file.name);
    setResults(null);
    setError(null);

    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (parsed) => {
        // Drop the Label column if present — it's ground truth, not a
        // feature, and the model was never trained to see it as input.
        const cleaned = parsed.data.map((row) => {
          const { Label, ...features } = row;
          return features;
        });
        setRows(cleaned);
      },
      error: (err) => setError(`Failed to parse CSV: ${err.message}`),
    });
  }

  async function runPrediction() {
    if (rows.length === 0) return;
    setRunning(true);
    setError(null);
    try {
      const data = await predictBatch(rows, "xgboost");
      setResults(data.results);
    } catch (err) {
      setError(
        err.response?.data?.error ||
          "Prediction failed. Check that both the Node backend and FastAPI service are running."
      );
    } finally {
      setRunning(false);
    }
  }

  const attackCount = results?.filter((r) => r.prediction === "ATTACK").length ?? 0;
  const benignCount = results?.filter((r) => r.prediction === "BENIGN").length ?? 0;

  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="font-display text-xl text-ink">Traffic Analysis</h1>
        <p className="text-muted text-sm mt-1">
          Upload a CSV of network flow records to classify traffic with the trained model.
        </p>
      </div>

      <div className="bg-surface border border-hairline rounded-md p-6">
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFile}
          className="hidden"
        />
        <button
          onClick={() => fileInputRef.current.click()}
          className="flex items-center gap-2 px-4 py-2 bg-raised border border-hairline rounded-md text-ink text-sm hover:border-info transition-colors"
        >
          <Upload size={16} strokeWidth={1.75} />
          {fileName || "Choose CSV file"}
        </button>

        {rows.length > 0 && (
          <div className="mt-4 flex items-center gap-4">
            <span className="text-muted text-sm font-mono">
              {rows.length.toLocaleString()} records loaded
            </span>
            <button
              onClick={runPrediction}
              disabled={running}
              className="flex items-center gap-2 px-4 py-2 bg-safe/15 text-safe border border-safe/40 rounded-md text-sm hover:bg-safe/25 transition-colors disabled:opacity-50"
            >
              <Play size={14} strokeWidth={2} />
              {running ? "Running prediction..." : "Run prediction"}
            </button>
          </div>
        )}

        {error && (
          <div className="mt-4 bg-alert/10 border border-alert/40 rounded-md p-3 text-alert text-sm">
            {error}
          </div>
        )}
      </div>

      {results && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-raised border border-hairline rounded-md p-4">
              <div className="text-muted text-xs">Total classified</div>
              <div className="font-mono text-ink text-xl mt-1">{results.length}</div>
            </div>
            <div className="bg-raised border border-hairline rounded-md p-4">
              <div className="text-muted text-xs">Benign</div>
              <div className="font-mono text-safe text-xl mt-1">{benignCount}</div>
            </div>
            <div className="bg-raised border border-hairline rounded-md p-4">
              <div className="text-muted text-xs">Attack</div>
              <div className="font-mono text-alert text-xl mt-1">{attackCount}</div>
            </div>
          </div>

          <div className="bg-surface border border-hairline rounded-md p-5">
            <h2 className="font-display text-sm text-ink mb-4">Results</h2>
            <ResultsTable results={results} />
          </div>
        </div>
      )}
    </div>
  );
}
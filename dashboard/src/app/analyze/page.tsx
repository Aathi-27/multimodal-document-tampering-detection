'use client';

import { useState, useCallback } from 'react';
import { apiClient, PredictionResult } from '@/lib/api';

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [earlyExit, setEarlyExit] = useState(true);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setError(null);
    setResult(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.predict(file, earlyExit);
      setResult(res);
    } catch (e: any) {
      setError(e.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const riskColor = (tier: string) => {
    if (tier === 'HIGH') return 'risk-high';
    if (tier === 'MEDIUM') return 'risk-medium';
    return 'risk-low';
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Analyze Document</h1>
        <p className="text-slate-400 mt-1">Upload a document to detect tampering</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div>
          <div
            className={`card border-2 border-dashed ${dragOver ? 'border-blue-500 bg-blue-900/10' : 'border-slate-600'} 
                        text-center p-8 transition-colors cursor-pointer`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input')?.click()}
          >
            <input
              id="file-input"
              type="file"
              accept="image/jpeg,image/png,image/jpg,application/pdf"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            
            {preview ? (
              <div>
                <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg mb-4" />
                <p className="text-sm text-slate-400">{file?.name}</p>
              </div>
            ) : (
              <div>
                <p className="text-4xl mb-4">📄</p>
                <p className="text-lg font-medium mb-2">Drop document here or click to upload</p>
                <p className="text-sm text-slate-400">Supports JPEG, PNG, PDF (max 50MB)</p>
              </div>
            )}
          </div>

          {/* Options */}
          <div className="card mt-4">
            <h3 className="font-semibold mb-3">Options</h3>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={earlyExit}
                onChange={(e) => setEarlyExit(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">Early exit (skip expensive stages when confident)</span>
            </label>
            <p className="text-xs text-slate-400 mt-1 ml-6">
              Reduces latency by 40-60% for clearly tampered/clean documents
            </p>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!file || loading}
            className="btn-primary w-full mt-4 py-3 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="animate-spin">⏳</span> Analyzing...
              </span>
            ) : (
              '🔍 Analyze Document'
            )}
          </button>

          {error && (
            <div className="card bg-red-900/30 border-red-700 mt-4">
              <p className="text-red-300">❌ {error}</p>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div>
          {result ? (
            <div className="space-y-4">
              {/* Risk Badge */}
              <div className="card text-center">
                <div className={`risk-badge ${riskColor(result.risk_tier)} text-2xl px-6 py-3 mb-3`}>
                  {result.risk_tier} RISK
                </div>
                <p className="text-3xl font-bold font-mono">{result.risk_score.toFixed(3)}</p>
                <p className="text-slate-400 text-sm mt-1">Fusion Risk Score</p>
              </div>

              {/* Signal Scores */}
              <div className="card">
                <h3 className="font-semibold mb-3">Signal Scores</h3>
                <div className="grid grid-cols-2 gap-4">
                  <ScoreCard label="Tamper Probability" value={result.tamper_probability} />
                  <ScoreCard label="MC Dropout Mean" value={result.mc_dropout_mean} />
                  <ScoreCard label="Uncertainty (σ)" value={result.mc_dropout_uncertainty} />
                  <ScoreCard label="OCR Tokens" value={result.ocr_token_count} isCount />
                </div>
              </div>

              {/* Explanations */}
              <div className="card">
                <h3 className="font-semibold mb-3">Explanations</h3>
                {result.explanations.length > 0 ? (
                  <ul className="space-y-2">
                    {result.explanations.map((exp, i) => (
                      <li key={i} className="text-sm bg-yellow-900/20 border-l-4 border-yellow-500 p-3 rounded-r-lg">
                        {exp}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-slate-400 text-sm">No significant anomalies detected.</p>
                )}
              </div>

              {/* Timing */}
              <div className="card">
                <h3 className="font-semibold mb-3">Timing Breakdown</h3>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  {Object.entries(result.timing_ms)
                    .filter(([_, v]) => typeof v === 'number')
                    .map(([key, val]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-slate-400">{key}</span>
                        <span className="font-mono">{val.toFixed(1)} ms</span>
                      </div>
                    ))}
                </div>
              </div>

              {/* Metadata */}
              <div className="card">
                <h3 className="font-semibold mb-3">Metadata</h3>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Document Type</span>
                    <span>{result.document_type || 'Unknown'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Request ID</span>
                    <span className="font-mono text-xs">{result.request_id.slice(0, 8)}...</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Timestamp</span>
                    <span className="text-xs">{new Date(result.timestamp).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="card h-96 flex items-center justify-center text-slate-400">
              <div className="text-center">
                <p className="text-4xl mb-4">📊</p>
                <p>Results will appear here after analysis</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ScoreCard({ label, value, isCount = false }: { label: string; value: number; isCount?: boolean }) {
  return (
    <div className="bg-slate-800 rounded-lg p-3 text-center">
      <p className="text-xl font-bold font-mono">
        {isCount ? value : value.toFixed(3)}
      </p>
      <p className="text-xs text-slate-400 mt-1">{label}</p>
    </div>
  );
}

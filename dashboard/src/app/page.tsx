'use client';

import { useEffect, useState } from 'react';
import { apiClient, StatsSummary, HealthStatus, ModelInfo } from '@/lib/api';

export default function DashboardPage() {
  const [stats, setStats] = useState<StatsSummary | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const [h, s, m] = await Promise.allSettled([
          apiClient.health(),
          apiClient.stats(),
          apiClient.modelInfo(),
        ]);
        if (h.status === 'fulfilled') setHealth(h.value);
        if (s.status === 'fulfilled') setStats(s.value);
        if (m.status === 'fulfilled') setModelInfo(m.value);
      } catch (e: any) {
        setError(e.message || 'Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-slate-400 mt-1">Overview of document tampering detection system</p>
      </div>

      {error && (
        <div className="card bg-red-900/30 border-red-700 mb-6">
          <p className="text-red-300">⚠️ {error}</p>
          <p className="text-sm text-slate-400 mt-1">Make sure the API server is running at the configured URL.</p>
        </div>
      )}

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="card">
          <p className="text-slate-400 text-sm">System Status</p>
          <p className={`text-2xl font-bold ${health?.status === 'healthy' ? 'text-green-400' : 'text-red-400'}`}>
            {health?.status === 'healthy' ? '✅ Healthy' : '❌ Down'}
          </p>
        </div>
        <div className="card">
          <p className="text-slate-400 text-sm">Total Predictions</p>
          <p className="text-2xl font-bold">{stats?.total_predictions ?? '—'}</p>
        </div>
        <div className="card">
          <p className="text-slate-400 text-sm">Mean Risk Score</p>
          <p className="text-2xl font-bold">
            {stats?.mean_risk_score != null ? stats.mean_risk_score.toFixed(3) : '—'}
          </p>
        </div>
        <div className="card">
          <p className="text-slate-400 text-sm">Alerts</p>
          <p className={`text-2xl font-bold ${stats?.total_alerts ? 'text-red-400' : 'text-green-400'}`}>
            {stats?.total_alerts ?? 0}
          </p>
        </div>
      </div>

      {/* Tier Distribution */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Risk Tier Distribution</h2>
          {stats?.tier_distribution ? (
            <div className="space-y-3">
              {Object.entries(stats.tier_distribution).map(([tier, pct]) => (
                <div key={tier}>
                  <div className="flex justify-between text-sm mb-1">
                    <span>{tier}</span>
                    <span>{(pct * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full ${
                        tier === 'HIGH' ? 'bg-red-500' : tier === 'MEDIUM' ? 'bg-yellow-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${pct * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-slate-400">No data yet</p>
          )}
        </div>

        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Performance Metrics</h2>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-slate-400">Mean Latency</span>
              <span className="font-mono">
                {stats?.mean_latency_ms ? `${stats.mean_latency_ms.toFixed(0)} ms` : '—'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">P95 Latency</span>
              <span className="font-mono">
                {stats?.p95_latency_ms ? `${stats.p95_latency_ms.toFixed(0)} ms` : '—'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Baseline Set</span>
              <span>{stats?.baseline_set ? '✅ Yes' : '❌ No'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Uptime</span>
              <span className="font-mono">
                {health?.uptime_seconds ? `${(health.uptime_seconds / 3600).toFixed(1)}h` : '—'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Info */}
      {modelInfo && (
        <div className="card mb-8">
          <h2 className="text-lg font-semibold mb-4">Model Information</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-slate-400 text-sm">Architecture</p>
              <p className="font-semibold">{modelInfo.architecture}</p>
            </div>
            <div>
              <p className="text-slate-400 text-sm">Input Shape</p>
              <p className="font-mono text-sm">{modelInfo.input_shape.join(' × ')}</p>
            </div>
            <div>
              <p className="text-slate-400 text-sm">Total Params</p>
              <p className="font-mono text-sm">{(modelInfo.total_params / 1e6).toFixed(1)}M</p>
            </div>
            <div>
              <p className="text-slate-400 text-sm">Framework</p>
              <p className="font-semibold">{modelInfo.framework}</p>
            </div>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="card">
        <h2 className="text-lg font-semibold mb-4">Quick Actions</h2>
        <div className="flex gap-4">
          <a href="/analyze" className="btn-primary">🔎 Analyze Document</a>
          <a href="/history" className="btn-secondary">📜 View History</a>
        </div>
      </div>
    </div>
  );
}

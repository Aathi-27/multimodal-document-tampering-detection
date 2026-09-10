'use client';

import { useState } from 'react';

export default function HistoryPage() {
  // In production, this would fetch from the API/database
  const [history] = useState<any[]>([]);

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Analysis History</h1>
        <p className="text-slate-400 mt-1">Previously analyzed documents and their results</p>
      </div>

      {history.length === 0 ? (
        <div className="card text-center py-16">
          <p className="text-4xl mb-4">📜</p>
          <p className="text-lg font-medium mb-2">No analysis history yet</p>
          <p className="text-slate-400 mb-4">Documents you analyze will appear here</p>
          <a href="/analyze" className="btn-primary">🔎 Analyze Your First Document</a>
        </div>
      ) : (
        <div className="space-y-4">
          {history.map((item, i) => (
            <div key={i} className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-semibold">{item.filename}</p>
                  <p className="text-sm text-slate-400">{new Date(item.timestamp).toLocaleString()}</p>
                </div>
                <div className={`risk-badge ${
                  item.risk_tier === 'HIGH' ? 'risk-high' : 
                  item.risk_tier === 'MEDIUM' ? 'risk-medium' : 'risk-low'
                }`}>
                  {item.risk_tier} — {item.risk_score.toFixed(3)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="card mt-8">
        <h2 className="text-lg font-semibold mb-3">💡 About History Storage</h2>
        <p className="text-slate-400 text-sm">
          In the demo configuration, analysis history is stored in browser session. 
          For production deployments, connect to a database (PostgreSQL, MongoDB) or 
          use S3 storage to persist history across sessions and users.
        </p>
      </div>
    </div>
  );
}

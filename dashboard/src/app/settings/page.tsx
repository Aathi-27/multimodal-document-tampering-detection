'use client';

import { useState } from 'react';

export default function SettingsPage() {
  const [apiUrl, setApiUrl] = useState(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');
  const [apiKey, setApiKey] = useState('');
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    localStorage.setItem('api_url', apiUrl);
    localStorage.setItem('api_key', apiKey);
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="text-slate-400 mt-1">Configure the dashboard and API connection</p>
      </div>

      <div className="max-w-2xl space-y-6">
        {/* API Configuration */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">🔌 API Configuration</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">API Base URL</label>
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-2 text-white"
                placeholder="http://localhost:8000"
              />
              <p className="text-xs text-slate-400 mt-1">URL of the FastAPI backend server</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">API Key (optional)</label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-2 text-white"
                placeholder="Leave empty if no authentication"
              />
              <p className="text-xs text-slate-400 mt-1">Required if API_KEY is set on the server</p>
            </div>
            <button onClick={handleSave} className="btn-primary">
              {saved ? '✅ Saved!' : '💾 Save Settings'}
            </button>
          </div>
        </div>

        {/* Fusion Weights */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">⚖️ Fusion Weight Profiles</h2>
          <div className="space-y-3">
            <WeightProfile
              name="Baseline"
              description="Balanced weights for general documents"
              weights={{ visual: 0.25, ocr_overlap: 0.30, ocr_conflict: 0.15, ocr_conf: 0.10, uncertainty: 0.10, spatial: 0.10 }}
            />
            <WeightProfile
              name="OCR Heavy"
              description="Emphasizes text-based signals for paystubs/statements"
              weights={{ visual: 0.15, ocr_overlap: 0.40, ocr_conflict: 0.20, ocr_conf: 0.10, uncertainty: 0.05, spatial: 0.10 }}
            />
            <WeightProfile
              name="Visual Heavy"
              description="Emphasizes visual forensics for IDs and passports"
              weights={{ visual: 0.35, ocr_overlap: 0.25, ocr_conflict: 0.10, ocr_conf: 0.05, uncertainty: 0.15, spatial: 0.10 }}
            />
          </div>
        </div>

        {/* Alert Configuration */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">🔔 Alert Channels</h2>
          <div className="space-y-3 text-sm">
            <AlertRow name="Email (AWS SES)" envVar="ALERT_EMAIL_ENABLED" />
            <AlertRow name="Slack Webhook" envVar="ALERT_SLACK_ENABLED" />
            <AlertRow name="SMS (Twilio)" envVar="ALERT_SMS_ENABLED" />
            <AlertRow name="Generic Webhook" envVar="ALERT_WEBHOOK_ENABLED" />
          </div>
          <p className="text-xs text-slate-400 mt-4">
            Alert channels are configured via environment variables on the API server.
            See <code className="bg-slate-800 px-1 rounded">alerting.py</code> for setup instructions.
          </p>
        </div>

        {/* System Info */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">ℹ️ System Information</h2>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-slate-400">Dashboard Version:</span>
              <span className="ml-2">2.0.0</span>
            </div>
            <div>
              <span className="text-slate-400">API Version:</span>
              <span className="ml-2">2.0.0</span>
            </div>
            <div>
              <span className="text-slate-400">Framework:</span>
              <span className="ml-2">Next.js 14 + React 18</span>
            </div>
            <div>
              <span className="text-slate-400">Backend:</span>
              <span className="ml-2">FastAPI + TensorFlow</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function WeightProfile({ name, description, weights }: {
  name: string;
  description: string;
  weights: Record<string, number>;
}) {
  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <div className="flex justify-between items-start mb-2">
        <div>
          <p className="font-medium">{name}</p>
          <p className="text-xs text-slate-400">{description}</p>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 text-xs">
        {Object.entries(weights).map(([key, val]) => (
          <div key={key} className="flex justify-between">
            <span className="text-slate-400">{key}:</span>
            <span className="font-mono">{val.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AlertRow({ name, envVar }: { name: string; envVar: string }) {
  return (
    <div className="flex justify-between items-center bg-slate-800 rounded-lg p-3">
      <div>
        <p className="font-medium">{name}</p>
        <p className="text-xs text-slate-400 font-mono">{envVar}</p>
      </div>
      <span className="text-xs bg-slate-700 px-2 py-1 rounded">Configure on server</span>
    </div>
  );
}

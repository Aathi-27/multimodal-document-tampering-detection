/**
 * API client for the Document Tampering Detection backend.
 * 
 * Communicates with the FastAPI server at the configured URL.
 * All methods return typed responses and handle errors gracefully.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface PredictionResult {
  request_id: string;
  risk_score: number;
  risk_tier: 'LOW' | 'MEDIUM' | 'HIGH';
  tamper_probability: number;
  mc_dropout_mean: number;
  mc_dropout_uncertainty: number;
  explanations: string[];
  ocr_token_count: number;
  document_type?: string;
  timing_ms: Record<string, number>;
  timestamp: string;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  version: string;
  uptime_seconds: number;
}

export interface BatchResult {
  request_id: string;
  filename: string;
  risk_score?: number;
  risk_tier?: string;
  tamper_probability?: number;
  document_type?: string;
  timing_ms?: Record<string, number>;
  status: 'success' | 'error';
  error?: string;
}

export interface ModelInfo {
  architecture: string;
  input_shape: number[];
  num_classes: number;
  classes: string[];
  total_params: number;
  trainable_params: number;
  framework: string;
}

export interface StatsSummary {
  total_predictions: number;
  mean_risk_score: number;
  median_risk_score: number;
  tier_distribution: Record<string, number>;
  mean_latency_ms: number | null;
  p95_latency_ms: number | null;
  baseline_set: boolean;
  total_alerts: number;
}

class ApiClient {
  private baseUrl: string;
  private apiKey: string;

  constructor(baseUrl?: string, apiKey?: string) {
    this.baseUrl = baseUrl || API_BASE;
    this.apiKey = apiKey || '';
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = {};
    if (this.apiKey) h['X-API-Key'] = this.apiKey;
    return h;
  }

  async health(): Promise<HealthStatus> {
    const res = await fetch(`${this.baseUrl}/health`, { headers: this.headers() });
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  }

  async predict(file: File, earlyExit: boolean = true): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append('file', file);

    const url = `${this.baseUrl}/predict?early_exit=${earlyExit}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: this.headers(),
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Prediction failed: ${res.statusText}`);
    }

    return res.json();
  }

  async predictBatch(files: File[], earlyExit: boolean = true): Promise<{ batch_results: BatchResult[]; total: number }> {
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));

    const url = `${this.baseUrl}/predict/batch?early_exit=${earlyExit}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: this.headers(),
      body: formData,
    });

    if (!res.ok) throw new Error(`Batch prediction failed: ${res.statusText}`);
    return res.json();
  }

  async modelInfo(): Promise<ModelInfo> {
    const res = await fetch(`${this.baseUrl}/model/info`, { headers: this.headers() });
    if (!res.ok) throw new Error(`Model info failed: ${res.statusText}`);
    return res.json();
  }

  async stats(): Promise<StatsSummary> {
    const res = await fetch(`${this.baseUrl}/stats`, { headers: this.headers() });
    if (!res.ok) throw new Error(`Stats failed: ${res.statusText}`);
    return res.json();
  }
}

export const apiClient = new ApiClient();
export default ApiClient;

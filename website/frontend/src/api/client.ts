import axios from 'axios';

// Same-origin by default (the web UI is served by the platform app);
// override with VITE_API_URL for local development.
const API_BASE = import.meta.env.VITE_API_URL || '';

export interface LeaderboardEntry {
  model_name: string;
  architecture: string;
  workload_type: string;
  best_latency_ms: number | null;
  best_throughput: number | null;
  best_power_watts: number | null;
  benchmark_count: number;
}

export interface ModelHistoryPoint {
  timestamp: string;
  results: {
    latency: { p50_ms: number };
    throughput: { samples_per_second: number };
    power: { average_watts: number | null };
  };
}

export const api = axios.create({ baseURL: API_BASE });

export async function fetchLeaderboard(
  metric = 'latency_p50',
  topK = 20
): Promise<LeaderboardEntry[]> {
  const { data } = await api.get('/api/leaderboard', {
    params: { metric, top_k: topK },
  });
  return data.leaderboard;
}

export async function fetchModelHistory(modelId: string): Promise<ModelHistoryPoint[]> {
  const { data } = await api.get(`/api/models/${encodeURIComponent(modelId)}/history`);
  return data.history;
}

export async function compareModels(modelIds: string[]) {
  const { data } = await api.post('/api/compare', { model_ids: modelIds });
  return data.comparison;
}

export interface ModelSubmission {
  model_id: string;
  name?: string;
  architecture?: string;
  parameters?: number;
  contact_email?: string;
  metrics?: Record<string, number | string>;
  notes?: string;
}

export async function submitModel(submission: ModelSubmission) {
  const { data } = await api.post('/api/submissions', submission);
  return data;
}

export interface BenchmarkRequest {
  model_id: string;
  workload_type?: string;
  precision?: string;
  batch_sizes?: number[];
  iterations?: number;
  warmup_iterations?: number;
  custom_config?: Record<string, unknown>;
}

export interface BenchmarkRun {
  status: string;
  run_id: string;
  device?: string;
  simulated?: boolean;
  hardware?: { gpu_name?: string | null; driver_version?: string | null };
  results?: {
    latency?: { p50_ms?: number; p95_ms?: number; p99_ms?: number };
    throughput?: { samples_per_second?: number };
    power?: { average_watts?: number };
    memory?: { peak_mb?: number };
    thermal?: { peak_temp_c?: number };
  };
}

export async function runBenchmark(request: BenchmarkRequest): Promise<BenchmarkRun> {
  const { data } = await api.post('/api/benchmark/run', request);
  return data;
}

export interface ToolInfo {
  name: string;
  description: string;
}

export async function fetchTools(): Promise<ToolInfo[]> {
  const { data } = await api.get('/api/tools');
  return data.tools;
}

export async function fetchHardware(): Promise<Record<string, unknown>> {
  const { data } = await api.get('/api/hardware');
  return data;
}

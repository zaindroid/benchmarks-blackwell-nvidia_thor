import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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

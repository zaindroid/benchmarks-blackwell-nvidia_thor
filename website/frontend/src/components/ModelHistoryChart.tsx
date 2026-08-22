import React, { useEffect, useState } from 'react';
import { fetchLeaderboard, fetchModelHistory, ModelHistoryPoint } from '../api/client';
import { BenchmarkChart } from './BenchmarkChart';

interface Point {
  timestamp: string;
  latency: number;
  throughput: number;
  power: number;
}

export const ModelHistoryChart: React.FC = () => {
  const [points, setPoints] = useState<Point[]>([]);
  const [model, setModel] = useState('');
  const [metric, setMetric] = useState<'latency' | 'throughput' | 'power'>('latency');

  useEffect(() => {
    fetchLeaderboard('latency_p50', 1)
      .then((entries) => {
        if (entries.length > 0) {
          setModel(entries[0].model_name);
          return fetchModelHistory(entries[0].model_name);
        }
        return [];
      })
      .then((history: ModelHistoryPoint[]) =>
        setPoints(
          history.map((h) => ({
            timestamp: h.timestamp,
            latency: h.results?.latency?.p50_ms ?? 0,
            throughput: h.results?.throughput?.samples_per_second ?? 0,
            power: h.results?.power?.average_watts ?? 0,
          }))
        )
      )
      .catch(() => setPoints([]));
  }, []);

  if (points.length === 0) {
    return null;
  }

  return (
    <div className="bg-white border rounded-lg p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">
          History: <span className="font-mono text-sm">{model}</span>
        </h3>
        <select
          value={metric}
          onChange={(e) => setMetric(e.target.value as typeof metric)}
          className="border rounded px-3 py-1 text-sm"
        >
          <option value="latency">Latency</option>
          <option value="throughput">Throughput</option>
          <option value="power">Power</option>
        </select>
      </div>
      <BenchmarkChart
        data={points}
        metric={metric}
        title={`${metric.charAt(0).toUpperCase() + metric.slice(1)} over time`}
      />
    </div>
  );
};

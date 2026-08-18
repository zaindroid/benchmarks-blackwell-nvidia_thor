import React, { useEffect, useState } from 'react';
import { fetchLeaderboard, LeaderboardEntry } from '../api/client';

export const Leaderboard: React.FC = () => {
  const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
  const [metric, setMetric] = useState('latency_p50');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchLeaderboard(metric)
      .then(setEntries)
      .catch((err) => console.error('Failed to fetch leaderboard:', err))
      .finally(() => setLoading(false));
  }, [metric]);

  if (loading) {
    return <div className="text-center py-8">Loading leaderboard...</div>;
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">ThorBench Leaderboard</h2>
        <select
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
          className="border rounded px-3 py-2"
        >
          <option value="latency_p50">Latency (P50)</option>
          <option value="throughput">Throughput</option>
          <option value="power_watts">Power Efficiency</option>
        </select>
      </div>

      {entries.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          No benchmark results yet. Run benchmarks with{' '}
          <code className="bg-gray-100 px-1">thor-benchmark run</code> and push them to the
          database.
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="min-w-full bg-white shadow-md rounded-lg">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-6 py-3 text-left">Rank</th>
              <th className="px-6 py-3 text-left">Model</th>
              <th className="px-6 py-3 text-left">Architecture</th>
              <th className="px-6 py-3 text-left">Workload</th>
              <th className="px-6 py-3 text-right">Latency (ms)</th>
              <th className="px-6 py-3 text-right">Throughput</th>
              <th className="px-6 py-3 text-right">Power (W)</th>
              <th className="px-6 py-3 text-right">Runs</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry, index) => (
              <tr key={entry.model_name} className="border-t hover:bg-gray-50">
                <td className="px-6 py-4">{index + 1}</td>
                <td className="px-6 py-4 font-medium">{entry.model_name}</td>
                <td className="px-6 py-4">{entry.architecture}</td>
                <td className="px-6 py-4">{entry.workload_type}</td>
                <td className="px-6 py-4 text-right">
                  {entry.best_latency_ms?.toFixed(2) ?? '-'}
                </td>
                <td className="px-6 py-4 text-right">
                  {entry.best_throughput?.toFixed(0) ?? '-'}
                </td>
                <td className="px-6 py-4 text-right">
                  {entry.best_power_watts?.toFixed(1) ?? '-'}
                </td>
                <td className="px-6 py-4 text-right">{entry.benchmark_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

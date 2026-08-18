import React, { useState } from 'react';
import { compareModels } from '../api/client';

interface ComparisonRow {
  model_name: string;
  min_latency_ms: number | null;
  max_throughput: number | null;
  min_power_watts: number | null;
  min_memory_mb: number | null;
}

export const ModelComparison: React.FC = () => {
  const [modelIds, setModelIds] = useState('ultralytics/yolov8n, meta-llama/Llama-3-8B');
  const [rows, setRows] = useState<ComparisonRow[]>([]);

  const run = async () => {
    try {
      const ids = modelIds.split(',').map((s) => s.trim()).filter(Boolean);
      const comparison = await compareModels(ids);
      setRows(comparison);
    } catch (err) {
      console.error('Comparison failed:', err);
    }
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Model Comparison</h3>
      <div className="flex gap-2 mb-4">
        <input
          value={modelIds}
          onChange={(e) => setModelIds(e.target.value)}
          placeholder="comma-separated model ids"
          className="flex-1 border rounded px-3 py-2"
        />
        <button
          onClick={run}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          Compare
        </button>
      </div>
      {rows.length > 0 && (
        <table className="min-w-full">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-4 py-2 text-left">Model</th>
              <th className="px-4 py-2 text-right">Latency (ms)</th>
              <th className="px-4 py-2 text-right">Throughput</th>
              <th className="px-4 py-2 text-right">Power (W)</th>
              <th className="px-4 py-2 text-right">Memory (MB)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.model_name} className="border-t">
                <td className="px-4 py-2">{row.model_name}</td>
                <td className="px-4 py-2 text-right">{row.min_latency_ms?.toFixed(2) ?? '-'}</td>
                <td className="px-4 py-2 text-right">{row.max_throughput?.toFixed(0) ?? '-'}</td>
                <td className="px-4 py-2 text-right">{row.min_power_watts?.toFixed(1) ?? '-'}</td>
                <td className="px-4 py-2 text-right">{row.min_memory_mb?.toFixed(1) ?? '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer
} from 'recharts';

interface BenchmarkData {
  timestamp: string;
  latency: number;
  throughput: number;
  power: number;
}

interface BenchmarkChartProps {
  data: BenchmarkData[];
  metric: 'latency' | 'throughput' | 'power';
  title: string;
}

const metricConfig = {
  latency: { color: '#8884d8', unit: 'ms', label: 'Latency' },
  throughput: { color: '#82ca9d', unit: 'samples/s', label: 'Throughput' },
  power: { color: '#ffc658', unit: 'W', label: 'Power' }
};

export const BenchmarkChart: React.FC<BenchmarkChartProps> = ({ data, metric, title }) => {
  const config = metricConfig[metric];

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" />
          <YAxis label={{ value: config.unit, angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey={metric}
            stroke={config.color}
            name={config.label}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

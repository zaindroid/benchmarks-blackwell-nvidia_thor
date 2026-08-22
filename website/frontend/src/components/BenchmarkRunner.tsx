import React, { useState } from 'react';
import { runBenchmark, BenchmarkRun } from '../api/client';

const ZOO_MODELS = [
  'ultralytics/yolov8n',
  'ultralytics/yolov8s',
  'ultralytics/yolov8m',
  'ultralytics/yolov8l',
  'hustvl/detr-resnet50',
  'nvidia/segformer-b0-finetuned-ade-512-512',
  'meta-llama/Llama-3-8B',
  'mistralai/Mistral-7B-v0.1',
  'microsoft/Phi-3-mini-4k-instruct',
  'llava-hf/llava-1.5-7b-hf',
];

const WORKLOADS = ['vision', 'language', 'multimodal', 'segmentation', 'classification'];
const PRECISIONS = ['fp32', 'fp16', 'int8', 'int4', 'fp8'];

const MetricCard: React.FC<{ label: string; value?: string | number | null; unit?: string }> = ({
  label,
  value,
  unit = '',
}) => (
  <div className="bg-gray-50 border rounded-lg p-4">
    <div className="text-sm text-gray-500">{label}</div>
    <div className="text-2xl font-bold mt-1">
      {value === null || value === undefined ? '—' : `${value}${unit}`}
    </div>
  </div>
);

export const BenchmarkRunner: React.FC = () => {
  const [modelId, setModelId] = useState('ultralytics/yolov8n');
  const [workload, setWorkload] = useState('vision');
  const [precision, setPrecision] = useState('fp16');
  const [batchSizes, setBatchSizes] = useState('1,4,8');
  const [iterations, setIterations] = useState('200');
  const [simulate, setSimulate] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<BenchmarkRun | null>(null);

  const submit = async () => {
    setRunning(true);
    setError('');
    setResult(null);
    try {
      const run = await runBenchmark({
        model_id: modelId.trim(),
        workload_type: workload,
        precision,
        batch_sizes: batchSizes
          .split(',')
          .map((s) => Number(s.trim()))
          .filter((n) => Number.isFinite(n) && n > 0),
        iterations: Number(iterations) || undefined,
        custom_config: simulate ? { simulate: true } : {},
      });
      setResult(run);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'Benchmark failed');
    } finally {
      setRunning(false);
    }
  };

  const r = result?.results;
  const simulated = result?.simulated ?? simulate;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Run a benchmark</h2>
        <p className="text-gray-600 mt-1">
          Executes on the connected NVIDIA DRIVE Thor device. Leave
          &quot;Simulate&quot; unchecked for real measurements with live
          power, memory and thermal sampling.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-4 bg-white border rounded-lg p-5">
        <label className="block">
          <span className="text-sm font-medium">Model</span>
          <input
            list="zoo-models"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="mt-1 w-full border rounded px-3 py-2 font-mono text-sm"
          />
          <datalist id="zoo-models">
            {ZOO_MODELS.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
        </label>

        <label className="block">
          <span className="text-sm font-medium">Workload</span>
          <select
            value={workload}
            onChange={(e) => setWorkload(e.target.value)}
            className="mt-1 w-full border rounded px-3 py-2"
          >
            {WORKLOADS.map((w) => (
              <option key={w} value={w}>
                {w}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="text-sm font-medium">Precision</span>
          <select
            value={precision}
            onChange={(e) => setPrecision(e.target.value)}
            className="mt-1 w-full border rounded px-3 py-2"
          >
            {PRECISIONS.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="text-sm font-medium">Batch sizes (comma separated)</span>
          <input
            value={batchSizes}
            onChange={(e) => setBatchSizes(e.target.value)}
            className="mt-1 w-full border rounded px-3 py-2"
          />
        </label>

        <label className="block">
          <span className="text-sm font-medium">Iterations</span>
          <input
            type="number"
            min={1}
            value={iterations}
            onChange={(e) => setIterations(e.target.value)}
            className="mt-1 w-full border rounded px-3 py-2"
          />
        </label>

        <label className="flex items-center gap-3 mt-6">
          <input
            type="checkbox"
            checked={simulate}
            onChange={(e) => setSimulate(e.target.checked)}
            className="h-4 w-4"
          />
          <span className="text-sm font-medium">Simulate (no GPU required, deterministic)</span>
        </label>
      </div>

      <button
        onClick={submit}
        disabled={running || !modelId.trim()}
        className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
      >
        {running ? 'Running...' : 'Run benchmark'}
      </button>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4">{error}</div>
      )}

      {result && (
        <div className="bg-white border rounded-lg p-5 space-y-4">
          <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm text-gray-600">
            <span>
              run: <span className="font-mono">{result.run_id}</span>
            </span>
            <span>
              device: <span className="font-medium">{result.device || 'local'}</span>
            </span>
            <span>
              simulated: <span className="font-medium">{String(simulated)}</span>
            </span>
            <span>gpu: {result.hardware?.gpu_name || '—'}</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricCard label="Latency P50 (ms)" value={r?.latency?.p50_ms} />
            <MetricCard label="Latency P95 (ms)" value={r?.latency?.p95_ms} />
            <MetricCard label="Throughput (samples/s)" value={r?.throughput?.samples_per_second} />
            <MetricCard label="Power (W)" value={r?.power?.average_watts} />
            <MetricCard label="Memory peak (MB)" value={r?.memory?.peak_mb} />
            <MetricCard label="Thermal peak (C)" value={r?.thermal?.peak_temp_c} />
          </div>
          <details>
            <summary className="text-sm text-gray-500 cursor-pointer">Full result JSON</summary>
            <pre className="mt-2 bg-gray-900 text-green-300 text-xs rounded p-3 overflow-auto max-h-64">
              {JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
};

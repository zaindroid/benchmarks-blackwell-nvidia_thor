import React, { useEffect, useState } from 'react';
import { fetchHardware, fetchTools, ToolInfo } from '../api/client';

const CLI_COMMANDS = [
  {
    title: 'Run a real benchmark on the device',
    code: './tools/scripts/benchmark-device.sh',
  },
  {
    title: 'Benchmark a specific model and precision',
    code: './tools/scripts/benchmark-device.sh --model meta-llama/Llama-3-8B --workload language --precision int8',
  },
  {
    title: 'Submit measured results to the leaderboard for review',
    code: './tools/scripts/benchmark-device.sh --submit',
  },
  {
    title: 'Benchmark from the CLI directly',
    code: 'thor-benchmark run --model ultralytics/yolov8n --workload vision --precision fp16 --batch-sizes 1,4,8 --iterations 300',
  },
  {
    title: 'Serve the MCP endpoint (stdio or HTTP)',
    code: 'thor-mcp --stdio\nthor-mcp --http-mcp --port 8000',
  },
];

const MCP_CLIENTS = [
  {
    name: 'Claude Desktop / Cursor',
    config: `{
  "mcpServers": {
    "thor": {
      "url": "https://thor-platform.zaindroid.me/mcp"
    }
  }
}`,
  },
  {
    name: 'Codex CLI',
    config: `[mcp_servers.thor]
url = "https://thor-platform.zaindroid.me/mcp"`,
  },
  {
    name: 'opencode',
    config: `{
  "mcp": {
    "thor": {
      "type": "remote",
      "url": "https://thor-platform.zaindroid.me/mcp"
    }
  }
}`,
  },
];

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <section className="bg-white border rounded-lg p-5 space-y-3">
    <h3 className="text-lg font-semibold">{title}</h3>
    {children}
  </section>
);

const CodeBlock: React.FC<{ code: string }> = ({ code }) => (
  <pre className="bg-gray-900 text-green-300 text-xs rounded p-3 overflow-auto">{code}</pre>
);

export const GettingStarted: React.FC = () => {
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [hardware, setHardware] = useState<Record<string, unknown>>({});

  useEffect(() => {
    fetchTools().then(setTools).catch(() => setTools([]));
    fetchHardware().then(setHardware).catch(() => setHardware({}));
  }, []);

  const hw = hardware as Record<string, unknown>;
  const hwStatus = (hw.status as string) || 'unknown';

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Getting started</h2>
        <p className="text-gray-600 mt-1">
          Benchmark AI models on the connected NVIDIA DRIVE Thor device — through the web UI,
          an MCP client, the Python client, or the CLI.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-white border rounded-lg p-5">
          <div className="text-2xl font-bold text-blue-600">1</div>
          <div className="font-semibold mt-1">Run a benchmark</div>
          <p className="text-sm text-gray-600 mt-1">
            Use the Run Benchmark tab, or connect any MCP client to the endpoint below.
          </p>
          <CodeBlock code={'https://thor-platform.zaindroid.me/mcp'} />
        </div>
        <div className="bg-white border rounded-lg p-5">
          <div className="text-2xl font-bold text-blue-600">2</div>
          <div className="font-semibold mt-1">Review results</div>
          <p className="text-sm text-gray-600 mt-1">
            Results are stored automatically and ranked on the Leaderboard tab.
          </p>
          <CodeBlock code={'GET /api/leaderboard'} />
        </div>
        <div className="bg-white border rounded-lg p-5">
          <div className="text-2xl font-bold text-blue-600">3</div>
          <div className="font-semibold mt-1">Submit your own</div>
          <p className="text-sm text-gray-600 mt-1">
            Measured on your hardware? Submit results for review via the form on the Leaderboard tab.
          </p>
          <CodeBlock code={'POST /api/submissions'} />
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <Section title="Platform status">
          <dl className="text-sm space-y-1">
            <div className="flex justify-between">
              <dt className="text-gray-500">Status</dt>
              <dd className="font-medium">{hwStatus}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">GPU</dt>
              <dd>{String(hw.gpu_name || '—')}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Driver</dt>
              <dd>{String(hw.driver_version || '—')}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">CUDA</dt>
              <dd>{String(hw.cuda_version || '—')}</dd>
            </div>
          </dl>
        </Section>

        <Section title="CLI commands">
          <div className="space-y-3">
            {CLI_COMMANDS.map((c) => (
              <div key={c.title}>
                <div className="text-sm font-medium mb-1">{c.title}</div>
                <CodeBlock code={c.code} />
              </div>
            ))}
          </div>
        </Section>

        <Section title="Connect from an MCP client">
          <div className="space-y-3">
            {MCP_CLIENTS.map((c) => (
              <div key={c.name}>
                <div className="text-sm font-medium mb-1">{c.name}</div>
                <CodeBlock code={c.config} />
              </div>
            ))}
          </div>
        </Section>

        <Section title="Available MCP tools">
          {tools.length === 0 ? (
            <p className="text-sm text-gray-500">Loading tools...</p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-500 border-b">
                  <th className="py-1 pr-3">Tool</th>
                  <th className="py-1">Description</th>
                </tr>
              </thead>
              <tbody>
                {tools.map((t) => (
                  <tr key={t.name} className="border-b border-gray-100">
                    <td className="py-1 pr-3 font-mono text-xs">{t.name}</td>
                    <td className="py-1 text-gray-600">{t.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>
      </div>
    </div>
  );
};

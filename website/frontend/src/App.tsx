import { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { BenchmarkRunner } from './components/BenchmarkRunner';
import { Leaderboard } from './components/Leaderboard';
import { ModelComparison } from './components/ModelComparison';
import { SubmissionForm } from './components/SubmissionForm';
import { ModelHistoryChart } from './components/ModelHistoryChart';
import { GettingStarted } from './components/GettingStarted';

type Tab = 'run' | 'leaderboard' | 'getting-started';

const TABS: { id: Tab; label: string }[] = [
  { id: 'run', label: 'Run Benchmark' },
  { id: 'leaderboard', label: 'Leaderboard' },
  { id: 'getting-started', label: 'Getting Started' },
];

function App() {
  const [tab, setTab] = useState<Tab>('run');

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-gray-900 text-white py-6">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold">ThorBench</h1>
          <p className="text-gray-300">
            Benchmarking platform for NVIDIA DRIVE Thor
          </p>
        </div>
      </header>

      <nav className="bg-white border-b">
        <div className="container mx-auto px-4 flex gap-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                tab === t.id
                  ? 'border-blue-600 text-blue-700'
                  : 'border-transparent text-gray-500 hover:text-gray-800'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </nav>

      <main className="container mx-auto px-4 py-8">
        {tab === 'run' && <BenchmarkRunner />}
        {tab === 'leaderboard' && (
          <div className="space-y-8">
            <Leaderboard />
            <ModelHistoryChart />
            <ModelComparison />
            <SubmissionForm />
          </div>
        )}
        {tab === 'getting-started' && <GettingStarted />}
      </main>

      <footer className="border-t py-6">
        <div className="container mx-auto px-4 text-sm text-gray-500">
          ThorBench — open benchmarking for NVIDIA DRIVE Thor.
          Endpoint: https://thor-platform.zaindroid.me/mcp
        </div>
      </footer>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);

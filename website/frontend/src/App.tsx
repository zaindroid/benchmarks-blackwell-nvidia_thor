import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Leaderboard } from './components/Leaderboard';
import { BenchmarkChart } from './components/BenchmarkChart';
import { ModelComparison } from './components/ModelComparison';
import { SubmissionForm } from './components/SubmissionForm';

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-gray-900 text-white py-6">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold">ThorBench</h1>
          <p className="text-gray-300">Benchmark leaderboard for NVIDIA DRIVE Thor</p>
        </div>
      </header>
      <main className="container mx-auto px-4 py-8 space-y-8">
        <Leaderboard />
        <BenchmarkChart data={[]} metric="latency" title="Latency over time" />
        <ModelComparison />
        <SubmissionForm />
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);

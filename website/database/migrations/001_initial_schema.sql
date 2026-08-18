-- ThorAI Platform initial schema (PostgreSQL 16+)
-- run_id uses TEXT to match platform ids ("run-<hex>"); the backend
-- storage layer (thor_mcp.storage) is compatible with this layout.

-- benchmark_runs table
CREATE TABLE IF NOT EXISTS benchmark_runs (
    run_id TEXT PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    hardware_info JSONB NOT NULL,
    model_info JSONB NOT NULL,
    workload_info JSONB NOT NULL,
    results JSONB NOT NULL,
    git_commit VARCHAR(40),
    environment JSONB,
    tags TEXT[],
    created_by VARCHAR(255)
);

-- models table
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    architecture TEXT,
    parameters BIGINT,
    source TEXT,
    license TEXT,
    last_benchmarked TIMESTAMP WITH TIME ZONE,
    best_metrics JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- optimization_profiles table
CREATE TABLE IF NOT EXISTS optimization_profiles (
    profile_id TEXT PRIMARY KEY,
    model_id TEXT REFERENCES models(model_id),
    optimization_type TEXT NOT NULL,
    precision TEXT,
    config JSONB,
    performance_gain JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- experiments table (for tracking research experiments)
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    hypothesis TEXT,
    config JSONB,
    results JSONB,
    metrics JSONB,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- datasets table (benchmark dataset registry)
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT,
    task TEXT,
    source TEXT,
    license TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- reports table (generated reports)
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY,
    benchmark_id TEXT REFERENCES benchmark_runs(run_id),
    format TEXT,
    template TEXT,
    content TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_model ON benchmark_runs ((model_info->>'name'));
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_timestamp ON benchmark_runs (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_workload ON benchmark_runs ((workload_info->>'type'));
CREATE INDEX IF NOT EXISTS idx_optimization_profiles_model ON optimization_profiles (model_id);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments (status);

-- Views for common queries
CREATE OR REPLACE VIEW model_best_performance AS
SELECT
    model_info->>'name' as model_name,
    workload_info->>'type' as workload_type,
    MIN((results->'latency'->>'p50_ms')::float) as best_latency_ms,
    MAX((results->'throughput'->>'samples_per_second')::float) as best_throughput,
    MIN((results->'power'->>'average_watts')::float) as best_power_watts
FROM benchmark_runs
GROUP BY model_info->>'name', workload_info->>'type';

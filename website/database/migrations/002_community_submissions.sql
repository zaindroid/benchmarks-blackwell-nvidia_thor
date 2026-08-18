-- Community model submissions (submission portal)
CREATE TABLE IF NOT EXISTS submissions (
    submission_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    name TEXT,
    architecture TEXT,
    parameters BIGINT,
    source TEXT,
    contact_email TEXT,
    metrics JSONB,
    notes TEXT,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected
    review_comment TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    reviewed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_submissions_status ON submissions (status);
CREATE INDEX IF NOT EXISTS idx_submissions_model ON submissions (model_id);

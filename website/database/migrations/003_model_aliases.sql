-- Model name normalization: map free-form/alternate names to a canonical model_id
-- (e.g. "yolov8n", "YOLOv8-n" -> "ultralytics/yolov8n") so benchmark queries and
-- the leaderboard can group results regardless of how a caller spelled the name.
CREATE TABLE IF NOT EXISTS model_aliases (
    alias TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_aliases_model_id ON model_aliases (model_id);

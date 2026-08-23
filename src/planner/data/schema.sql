-- SQLite schema for Planner benchmark data

CREATE TABLE IF NOT EXISTS exported_summaries (
    id TEXT NOT NULL PRIMARY KEY,
    config_id TEXT NOT NULL,
    model_hf_repo TEXT NOT NULL,
    provider TEXT,
    type TEXT NOT NULL,
    ttft_mean REAL NOT NULL,
    ttft_p90 REAL NOT NULL,
    ttft_p95 REAL NOT NULL,
    ttft_p99 REAL NOT NULL,
    e2e_mean REAL NOT NULL,
    e2e_p90 REAL NOT NULL,
    e2e_p95 REAL NOT NULL,
    e2e_p99 REAL NOT NULL,
    itl_mean REAL,
    itl_p90 REAL,
    itl_p95 REAL,
    itl_p99 REAL,
    tps_mean REAL,
    tps_p90 REAL,
    tps_p95 REAL,
    tps_p99 REAL,
    hardware TEXT,
    hardware_count INTEGER,
    framework TEXT,
    requests_per_second REAL NOT NULL,
    responses_per_second REAL,
    tokens_per_second REAL NOT NULL,
    mean_input_tokens REAL NOT NULL,
    mean_output_tokens REAL NOT NULL,
    huggingface_prompt_dataset TEXT,
    jbenchmark_created_at TEXT NOT NULL,
    entrypoint TEXT,
    docker_image TEXT,
    framework_version TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    loaded_at TEXT,
    prompt_tokens INTEGER,
    prompt_tokens_stdev INTEGER,
    prompt_tokens_min INTEGER,
    prompt_tokens_max INTEGER,
    output_tokens INTEGER,
    output_tokens_min INTEGER,
    output_tokens_max INTEGER,
    output_tokens_stdev INTEGER,
    profiler_type TEXT,
    profiler_image TEXT,
    profiler_tag TEXT,
    source TEXT NOT NULL DEFAULT 'local',
    model_uri TEXT,
    confidence_level TEXT NOT NULL DEFAULT 'estimated'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_config_id_unique ON exported_summaries (config_id);

CREATE INDEX IF NOT EXISTS idx_benchmark_lookup
ON exported_summaries(model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens);

CREATE INDEX IF NOT EXISTS idx_traffic_patterns
ON exported_summaries(prompt_tokens, output_tokens);

CREATE INDEX IF NOT EXISTS idx_model_hardware
ON exported_summaries(model_hf_repo, hardware);

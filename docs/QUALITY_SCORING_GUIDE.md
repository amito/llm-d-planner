# Quality Scoring Guide

Planner compares LLM models by normalizing scores from multiple benchmark sources onto a common scale, then compositing them into a single rating. This guide explains how each piece works.

## Data Sources

Planner uses two independent data sources that measure model quality in fundamentally different ways.

### Arena (Human Preference)

[Arena](https://lmarena.ai/) (formerly LMSYS Chatbot Arena) ranks models using **human preference judgments**. Real users interact with pairs of anonymous models and vote for the one they prefer. Votes are aggregated into Bradley-Terry Elo ratings.

- **Score range**: ~700–1550 (Elo-style ratings)
- **Categories**: 27, including general (coding, math, creative writing, instruction following) and industry-specific (legal, science, software/IT, healthcare, etc.), plus 8 language-specific categories
- **Confidence intervals**: Each rating has lower/upper bounds reflecting statistical uncertainty
- **Strengths**: Captures subjective qualities — helpfulness, writing style, conversational fluency
- **Limitations**: Biased toward user-facing chat tasks; less coverage of specialized capabilities

Data is fetched from the HuggingFace dataset `lmarena-ai/leaderboard-dataset` and cached in `src/quality_scoring/data/arena_models.json`.

### Artificial Analysis (Automated Benchmarks)

[Artificial Analysis](https://artificialanalysis.ai/) evaluates models using **automated benchmark suites** — standardized tests with known correct answers, run programmatically.

- **Score range**: 0–100 for aggregate indices
- **Aggregate indices**: Intelligence Index (overall), Coding Index, Agentic Index
- **Intelligence Index (v4.0+)** aggregates scores from 10 evaluations: GPQA Diamond (science), Humanity's Last Exam (cross-domain), SciCode (scientific coding), Terminal-Bench Hard (CLI tasks), IFBench (instruction following), AA-LCR (long-context retrieval), AA-Omniscience (broad knowledge), GDPval-AA (quantitative reasoning), tau2-Bench Telecom (agent tasks), CritPt (critical thinking). See [AA's methodology](https://artificialanalysis.ai/methodology) for the current composition — AA may update this list over time.
- **Strengths**: Precise for measurable capabilities; reproducible; covers specific skill areas
- **Limitations**: Less reflective of subjective qualities like writing style or helpfulness

Data is sourced from the free V2 API (`/api/v2/language/models/free`) and cached in `src/quality_scoring/data/aa_models.json` (requires `AA_API_KEY`).

## Model Name Resolution

Models are identified differently across sources. Arena uses names like `llama-3.1-8b-instruct`, AA uses names like `Llama 3.1 Instruct 8B`, and deployment catalogs use HuggingFace repo IDs like `meta-llama/llama-3.1-8b-instruct`. The resolver matches these automatically.

### Match Types

- **Exact**: Identical string match
- **Equivalent**: Same model, different formatting (case, punctuation, word order, org prefix, separators)
- **Fuzzy**: Approximate match requiring `--fuzzy` flag to accept (suffix-stripped variants, subset tokens, version-adjacent, size-aware)

### Resolution Pipeline

The resolver tries 13 strategies in order, returning the first match:

| Step | Type | What it does |
|---|---|---|
| 1 | Exact | String equality |
| 2 | Equivalent | Case-insensitive |
| 3–4 | Equivalent | Org-prefix stripping (handles `meta-llama/`, `redhatai/`, and baked-in prefixes like `meta-`, `ibm-`, `nvidia-`, `amazon-`) |
| 5 | Equivalent | Punctuation normalization (dashes, underscores, spaces) |
| 6 | Equivalent | Separator normalization (dots, alpha-digit boundaries: `qwen2.5` = `qwen-2-5`) |
| 7 | Equivalent | Token-set matching (order-independent: `qwen2.5-coder-32b-instruct` = `qwen2.5 coder instruct 32b`) |
| 8–9 | Fuzzy | Suffix stripping (removes quantization, instruct, reasoning suffixes) |
| 10 | Fuzzy | Subset token matching (`Llama 4 Scout` tokens are a subset of `llama-4-scout-17b-16e-instruct`) |
| 11 | Fuzzy | Size-aware word matching (prefers candidates with matching parameter size) |
| 12 | Fuzzy | Version-adjacent (closest version number with same base) |
| 13 | Fuzzy | Normalized substring |

Fuzzy steps 10, 11, and 13 require the **model family name** to match. Family names are extracted as the first alphabetic token after org stripping (e.g., `llama`, `qwen`, `mistral`, `granite`). This prevents cross-family false matches like `qwen-7b-instruct` matching `mistral-7b-instruct`. Steps 8–9 (suffix-stripped matching) have no family check. Step 12 (version-adjacent) uses parsed version base comparison instead of family extraction.

### Token Filtering

Token-set matching filters out noise tokens to improve matching:
- **Quantization tokens**: `fp8`, `dynamic`, `nvfp4`, `hf`, and any token starting with `quantized` (e.g., `quantized.w4a16`, `quantized.w8a8`)
- **Date suffixes**: YYMM patterns for years 2024–2029 (e.g., `2501`, `2903`)
- **Numeric token ordering**: Pure digit tokens preserve their original order (`4-6` does not match `6-4`), while other tokens are order-independent

### Org Prefix Stripping

Some model names have vendor org names baked in with a dash separator instead of a slash. The resolver strips these known prefixes:
- `meta-` (e.g., `meta-llama-3.1-8b` → `llama-3.1-8b`)
- `ibm-` (e.g., `ibm-granite-3.1-8b` → `granite-3.1-8b`)
- `nvidia-` (e.g., `nvidia-nemotron-3-nano` → `nemotron-3-nano`)
- `amazon-` (e.g., `amazon-nova-pro` → `nova-pro`)

This list requires periodic review when new models or vendors appear. Check Arena/AA data and deployment catalogs for model names starting with vendor prefixes after each data sync.

## Score Normalization

Arena and AA use incompatible score ranges with very different distributions:
- Arena Elo ratings span ~700–1550 with mild top-compression
- AA intelligence/coding indices are 0–100 but severely bottom-compressed (95% of models score below 50)

### Tied-Rank Percentile

Scores are normalized to percentile ranks — "what percentage of models does this one beat?" — using a tied-rank approach:

1. **Sort** all models in a source/category by raw score (descending)
2. **Group** adjacent models whose scores are within the tie threshold into tied ranks
3. **Assign** each group the same percentile using the mean-rank method

A percentile of 85.0 means "this model scores higher than 85% of all models in this source."

### Tie Thresholds

Tie grouping prevents inflating meaningless score differences:

- **Arena**: Two models are tied if their confidence intervals overlap bidirectionally (`A.upper >= B.lower AND B.upper >= A.lower`). Uses the anchor model's CI — each new model is compared against the group's first (highest-rated) model, not the previous one.
- **AA**: Two models are tied if their scores are within `0.1 × standard deviation` of the group anchor.

### Population-Level Normalization

Percentiles are computed against the **full cached population** (all models in the source), not just the models being queried. This ensures stable, meaningful percentiles regardless of which models you compare.

## Composite Scoring

When both sources have data for a model in a category, the composite score is a weighted average of the two percentile ranks:

```
composite = arena_weight × arena_percentile + aa_weight × aa_percentile
```

Default source weights are equal (`arena_weight=1, aa_weight=1` in the ScoringEngine constructor). Per-category weights for each use case are defined in `src/planner/data/configuration/quality_weights.json`. Values are pure weights, normalized internally. When only one source has data, the composite equals that source's percentile.

### Provenance Flags

Each score carries a provenance flag indicating which sources contributed. The composite scores table uses short codes:
- `[B]` — Both sources (composited)
- `[A]` — Arena only
- `[AA]` — AA only

Category findings use full-word labels: `[Both]`, `[Arena Only]`, `[AA Only]`, `[Mixed]`. The "Mixed" label appears when some models in a category have both sources and others have only one.

## Tiers and Gap Significance

### Tier Classification

Models are classified into tiers for quick positioning. Two tier systems are used depending on context:

**Rank-based tiers** (used in per-source Key Findings):

| Tier | Rank Range |
|---|---|
| Frontier | 1–10 |
| Near-frontier | 11–25 |
| Upper-mid | 26–75 |
| Mid-tier | 76–150 |
| Long-tail | 151+ |

**Percentile-based tiers** (used in Composite Scores and Category Analysis):

| Tier | Percentile Range |
|---|---|
| Frontier | ≥95th |
| Near-frontier | 85th–94th |
| Upper-mid | 50th–84th |
| Mid-tier | 15th–49th |
| Long-tail | Below 15th |

### Gap Significance

Gap descriptions vary by context:

**Composite percentile gaps** (used in Category Analysis findings):

| Gap | Description |
|---|---|
| <5 percentile points | Effectively equivalent |
| 5–15 percentile points | Moderate advantage |
| >15 percentile points | Clear separation |

**Arena gaps** (per-source findings): Based on confidence interval overlap — "statistically indistinguishable" (CIs overlap), "small but statistically significant difference" (CIs don't overlap, gap <20 points), or "clear separation" (gap ≥20 points).

**AA gaps** (per-source findings): Based on gap relative to population standard deviation — "not clearly distinguishable" (<0.5σ), "moderate difference" (0.5–1.0σ), or "clear separation" (>1.0σ).

## Variant Handling

When the resolver matches a model variant (quantized, instruct, reasoning) to a different variant's benchmark score, the scoring pipeline applies an adjustment and marks it with reduced confidence.

### Quantization Discounts

Applied as a multiplicative factor to the raw score:

| Quantization | Factor | Quality Impact |
|---|---|---|
| FP8, FP8-dynamic | ×1.0 | Negligible |
| W8A8 | ×0.97 | Minor (3%) |
| W4A16, NVFP4 | ×0.92 | Moderate (8%) |

These factors are from empirical measurements in llm-d-planner.

### Reasoning/Instruct Variant Flagging

When the resolver matches across variant boundaries (e.g., a reasoning or thinking model matched to a non-reasoning variant's scores, or an instruct model matched to a base variant), the system flags the mismatch descriptively but does **not** apply a numerical adjustment. The variant detection matches both "reasoning" and "thinking" keywords. The variant description appears in confidence indicators.

The codebase includes functions to compute empirical reasoning deltas from paired models (`compute_reasoning_deltas()`, `compute_arena_thinking_delta()` in `variants.py`), but these are not currently wired into the scoring pipeline.

### Bidirectional Adjustments

Quantization adjustments work in both directions:
- Have full-precision, want quantized → multiply by discount
- Have quantized, want full-precision → divide by discount

### Confidence Indicators

Adjusted scores display `*` next to the percentile and a footnote explaining the adjustment.

## Missing Category Handling

When a model lacks scores for a specific category (e.g., a model has an overall score but no coding-specific score), the scoring engine applies a **discounted fallback** to the overall percentile.

### Current Approach

- **Fallback**: Use 0.8× the model's overall composite percentile for missing categories
- **Rationale**: Correlation analysis shows coding r=0.976 with overall, and all Arena categories show r>0.93. 93% of top-quartile-overall models are also top-quartile coding; 0% fall in bottom-quartile coding. The 0.8× discount provides fairness to models with measured category-specific scores, not correction of a prediction error.
- **Coverage**: Intelligence Index (overall) has 98% coverage, Coding Index 38%, Agentic Index 28%. The fallback ensures all models can be scored while incentivizing measurement.

This approach was chosen over alternatives like peer-minimum estimation or pre-computed fill-ins for its simplicity and empirical grounding in the correlation structure of the data.

## Planner Integration

Planner's quality scoring system integrates the dual-source ScoringEngine into the recommendation pipeline.

### Quality Weights Configuration

Per-use-case category weights are defined in `src/planner/data/configuration/quality_weights.json`. Each use case maps to integer weights for categories like `overall`, `coding`, `math`, `creative_writing`, etc.

Example:

```json
{
  "code_completion": {
    "categories": {
      "coding": 5,
      "math": 3,
      "overall": 2,
      "agentic": 2,
      "hard_prompts": 2
    }
  },
  "chatbot_conversational": {
    "categories": {
      "overall": 4,
      "instruction_following": 3,
      "multi_turn": 3,
      "creative_writing": 2,
      "hard_prompts": 2
    }
  }
}
```

Every use case includes `overall` to ensure at least one high-coverage dual-source signal. Weights are normalized internally — only relative proportions matter. The scoring engine computes a weighted composite percentile across all specified categories.

### Scoring Pipeline

1. **ScoringEngine initialization**: `build_scoring_engine()` in `src/planner/recommendation/quality/scoring.py` constructs a `ScoringEngine` from cached data in `src/quality_scoring/data/` (or auto-updates from `.quality_cache/` if `QUALITY_AUTO_UPDATE=true`).

2. **Per-model scoring**: `ConfigFinder` calls `engine.get_scores(model_hf_repo, fuzzy=True)` for each benchmark model, enabling fuzzy matching so that quantized variants (e.g., `redhatai/llama-3.3-70b-instruct-fp8-dynamic`) resolve to their base model's quality scores. `compute_quality_score(scorecard, category_weights)` then takes the resulting `ModelScorecard` and a dict of category weights, applies the per-category weights (with 0.8× fallback discount for missing categories), and returns a single float percentile (0–100).

3. **Integration with ConfigurationScores**: The quality score becomes `ConfigurationScores.quality_score`, replacing the old `accuracy_score`. This score is used in multi-criteria ranking alongside `price_score`, `latency_score`, and `complexity_score`.

### Hybrid Caching Strategy

Planner uses a two-tier cache:

1. **Checked-in snapshots** (`src/quality_scoring/data/`): Committed to git, provide stable baseline data for offline use and CI/CD. Updated manually via `make quality-sync`.

2. **Runtime auto-update cache** (`.quality_cache/`, gitignored): When `QUALITY_AUTO_UPDATE=true`, the scoring engine fetches fresh data from Arena (HuggingFace) and AA (free V2 API) on first use and stores it in `.quality_cache/`. Subsequent runs use the cached data until it expires (24-hour TTL). This keeps recommendations current without manual intervention.

Environment variables:
- `QUALITY_AUTO_UPDATE`: Enable/disable auto-update (default: `false` for stability)
- `AA_API_KEY`: Artificial Analysis API key (required for AA data sync)
- `LLM_QUALITY_CACHE_DIR`: Override cache directory (default: `.quality_cache/`)

### API Endpoints

Planner exposes quality data management via REST API:

- **GET /api/v1/quality/auto-update**: Returns current auto-update status and cache stats
- **PUT /api/v1/quality/auto-update**: Enable/disable auto-update at runtime
- **POST /api/v1/quality/refresh**: Manually trigger a quality data refresh (bypass TTL)

### Make Target

```bash
make quality-sync
```

Fetches fresh data from Arena (HuggingFace) and AA (free V2 API), updates `src/quality_scoring/data/` snapshots, and commits the changes. Run this periodically (e.g., weekly) to keep the baseline cache current. Requires `AA_API_KEY` environment variable.

### Checking Model Name Resolution

The `scripts/check_model_resolution.py` script checks how model names from a catalog (or a comma-separated list) resolve against Arena and AA data sources. Use it to verify resolver coverage after adding models to the catalog or after a data sync.

**Check the full model catalog (with fuzzy matching, as planner uses):**

```bash
uv run python scripts/check_model_resolution.py --catalog src/planner/data/configuration/model_catalog.json --fuzzy
```

**Check specific models:**

```bash
uv run python scripts/check_model_resolution.py -m "meta-llama/llama-3.1-8b-instruct,qwen/qwen2.5-72b-instruct"
```

**Flags:**

- `--catalog PATH` or `-m NAMES`: Mutually exclusive; provide either a catalog JSON file or comma-separated model names
- `--fuzzy`: Accept fuzzy matches (without this flag, fuzzy matches are shown but flagged as "not accepted")
- `--cache-dir PATH`: Override cache directory; falls back to bundled snapshots if the cache is empty

The output shows each model's resolution status per source (exact, equivalent, fuzzy, or not found) with similar-name suggestions for unmatched models, plus a summary of resolution coverage.

## ScoringEngine API

The `ScoringEngine` class in `src/quality_scoring/engine.py` provides the primary programmatic API for external consumers. It pre-computes normalizations across the full population once, then supports cheap per-model lookups via `get_scores()` and `get_scores_batch()`. The Planner recommendation engine delegates to it internally.

### Key Methods

- `get_scores(model_name, *, fuzzy=False)`: Returns a `ModelScorecard | None` with percentile scores for each category.
- `get_scores_batch(model_names, *, fuzzy=False)`: Batch version for multiple models, returns `list[ModelScorecard]`.

See the [quality-scoring package README](../src/quality_scoring/README.md) for full API documentation.

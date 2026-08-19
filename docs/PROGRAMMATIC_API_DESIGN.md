# Design: Programmatic API

## Problem

The current API surface grew organically around the Streamlit UI's needs. It mixes concerns, bundles steps that should be separate, and requires programmatic users to manually assemble data from multiple endpoints. The end-to-end flow — from user intent to deployed service — should be a clean pipeline of composable endpoints, each with a clear input/output contract.

### Current issues

1. **Intent extraction is bundled with the only pipeline entry point.** `POST /api/v1/extract` requires an LLM. Users who want to construct intent programmatically (or don't have an LLM available) have no path forward.

2. **No specification generation endpoint.** To build a complete specification, a programmatic user must call 3 separate GET endpoints (`/slo-defaults`, `/workload-profile`, `/expected-rps`), look up 2 JSON config files (`quality_weights.json`, `priority_weights.json`), and manually assemble the result.

3. **The recommendation endpoint has a flat request model.** `RankedRecommendationFromSpecRequest` scatters specification fields across 15+ top-level fields rather than accepting a structured `DeploymentSpecification`. It also hardcodes `domain_specialization` to `["general"]` and doesn't carry quality weights or priorities — those are baked into the backend.

4. **Deployment file generation and cluster deployment are coupled.** `POST /deploy` generates YAML and returns it inline. `POST /deploy-to-cluster` generates YAML AND applies it. There's no way to just get the deployment files, review them, modify them, and then deploy separately.

5. **`experience_class` is dead weight.** It's a 1:1 deterministic mapping from `use_case`, never set independently, never consumed by any downstream engine — but it's on the schema and inferred in 3 places.

6. **SLO default values are hardcoded to the 75th percentile** of the min/max range regardless of latency priority. The GUI currently prevents users from entering values outside the recommended range — this is a regression; users should be able to enter any value.

7. **API objects are loosely defined.** Endpoints use ad-hoc dicts and flat field lists instead of named, well-defined Pydantic models. Each stage's output should be a structured object that can be passed directly to the next stage.

## Design principles

1. **API surface, not engine.** This redesign reshapes the API layer — endpoint contracts, request/response models, and endpoint organization — without changing the underlying recommendation logic, scoring, or deployment generation.

2. **Composable pipeline.** Each endpoint's output is a named, well-defined object that can be passed directly as input to the next stage. Users can enter or exit the pipeline at any point.

2. **Named objects with field references.** Every API input and output is a named Pydantic model with documented fields, types, defaults, and valid values. Defaults are filled in so objects are always complete and ready to pass downstream.

3. **No required ordering.** Each endpoint is independently useful. A user can call any single endpoint in isolation by constructing the input themselves.

4. **Future: express path.** A future enhancement could allow skipping intermediate steps (e.g., `POST /generate-specification` with a flag to automatically feed the result to `/generate-recommendations`). Not designed here — saved for a future PR.

## Pipeline overview

```
extract-intent → generate-specification → generate-recommendations → generate-deployment → deploy-bundle-to-cluster
```

Each stage is independent. A user can:
- Enter at any stage by constructing the input object
- Exit at any stage with the output object
- Skip stages entirely
- Modify any object between stages

## Named objects

Each stage takes and returns a named object. All fields are populated with defaults so objects can be passed directly to the next stage without modification.

| Object | Produced by | Consumed by |
|---|---|---|
| `DeploymentIntent` | `extract-intent` (or user-constructed) | `generate-specification` |
| `DeploymentSpecification` | `generate-specification` (or user-constructed) | `generate-recommendations` |
| `RankedRecommendations` | `generate-recommendations` | User selects one `DeploymentRecommendation` |
| `DeploymentConfiguration` | Selected from `RankedRecommendations` | `generate-deployment` |
| `DeploymentRecommendation` | Selected from `RankedRecommendations` | User review |
| `DeploymentBundle` | `generate-deployment` | `deploy-bundle-to-cluster` |

## Object field references

### `DeploymentIntent`

The input to `generate-specification` and the output of `extract-intent`.

#### Required fields

| Field | Type | Valid values |
|---|---|---|
| `use_case` | string | `chatbot_conversational`, `code_completion`, `code_generation_detailed`, `translation`, `content_generation`, `summarization_short`, `document_analysis_rag`, `long_document_summarization`, `research_legal_analysis` |
| `user_count` | integer | Any positive integer |

#### Optional fields with constrained values

| Field | Type | Default | Valid values |
|---|---|---|---|
| `quality_priority` | string | `"medium"` | `low`, `medium`, `high` |
| `cost_priority` | string | `"medium"` | `low`, `medium`, `high` |
| `latency_priority` | string | `"medium"` | `low`, `medium`, `high` |

#### Optional fields with known values (open-ended)

| Field | Type | Default | Known values | Notes |
|---|---|---|---|---|
| `preferred_gpu_types` | list | `[]` | `L4`, `A10G`, `A100-40`, `A100-80`, `H100`, `H200`, `B200`, `MI300X`, `L40`, `L20`, `B100` | Normalized via `gpu_normalizer.py`; aliases accepted (e.g., `NVIDIA-H100`). Unknown values are skipped with a warning. Empty list = no GPU preference. |
| `preferred_models` | list | `[]` | 47 models in catalog | HuggingFace format (e.g., `meta-llama/Llama-3.1-8B-Instruct`). Can be catalog models or arbitrary HF repo IDs. Empty list = no model preference. |
| `domain_specialization` | list | `["general"]` | `general`, `code`, `enterprise`, `multilingual`, `reasoning`, `vision` | Not currently used by the recommendation engine, but will be used in the future to influence quality benchmark weight selection. |

#### GPU count limits

Each entry in `preferred_gpu_types` can optionally include a maximum GPU count:

```json
{
  "preferred_gpu_types": [
    "L4",
    {"gpu_type": "H100", "max_count": 4},
    {"gpu_type": "H200", "max_count": 2}
  ]
}
```

Plain strings mean no GPU count limit. Objects with `max_count` set a ceiling — configurations requiring more GPUs of that type are filtered out.

#### Fields removed

| Field | Reason |
|---|---|
| `experience_class` | Was a 1:1 deterministic mapping from `use_case`, never set independently, never consumed by any downstream engine. If needed in the future, derive on the fly from `use_case`. |

### `DeploymentSpecification`

The output of `generate-specification` and the input to `generate-recommendations`. Contains 4 sections plus the original intent.

#### Fields

| Field | Type | Description |
|---|---|---|
| `intent` | `DeploymentIntent` | The original intent (echoed back with defaults filled in) |
| `slo_targets` | `SLOTargets` | Latency targets at a given percentile |
| `workload_profile` | `WorkloadProfile` | Token counts, expected QPS |
| `quality_weights` | `QualityWeights` | Per-use-case category weights for quality scoring |
| `priorities` | `Priorities` | Quality/cost/latency priority levels and resolved weights |

#### `SLOTargets` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `ttft_target_ms` | integer | Derived from use case + latency priority | Time to First Token target. User can enter any positive value; the default is within the recommended range. |
| `itl_target_ms` | integer | Derived from use case + latency priority | Inter-Token Latency target |
| `e2e_target_ms` | integer | Derived from use case + latency priority | End-to-end latency target |
| `percentile` | string | `"p95"` | Which percentile the targets apply to. Valid values: `p90`, `p95`, `p99`. **Note:** only `p95` is currently supported by the backend benchmark data. `p90` and `p99` are accepted on the schema but will return an error until backend support is added. |
| `ttft_range` | object | From use case template | `{"min": int, "max": int}` — the recommended range. Informational; not enforced. |
| `itl_range` | object | From use case template | `{"min": int, "max": int}` — the recommended range. Informational; not enforced. |
| `e2e_range` | object | From use case template | `{"min": int, "max": int}` — the recommended range. Informational; not enforced. |

**SLO default value selection:** The default target value is derived from the recommended range using the `latency_priority` from the intent:

| Latency priority | Range percentile | Effect |
|---|---|---|
| `high` | 25th percentile | Tighter target, closer to min (aggressive) |
| `medium` | 50th percentile | Middle of range (balanced) |
| `low` | 75th percentile | Relaxed target, closer to max (permissive) |

Formula: `default = min + (max - min) * percentile`, rounded to nearest 5.

Example for `chatbot_conversational` (TTFT range 100–500ms):
- High priority → 200ms (25th percentile)
- Medium priority → 300ms (50th percentile)
- Low priority → 400ms (75th percentile)

This replaces the current dual approach where `_adjust_slo_for_latency()` applies multipliers (0.9x/1.0x/1.2x) to template base values. The range-based approach stays within defined bounds and is used consistently by both the API and the GUI.

**GUI behavior:** The GUI shows the recommended range but does NOT restrict input to that range. Users can enter any positive value. This is a fix for a regression — the GUI currently enforces the range as min/max on the input widget.

#### `WorkloadProfile` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt_tokens` | integer | From use case template | Mean input token length per request (GuideLLM traffic profile) |
| `output_tokens` | integer | From use case template | Mean output token length per request |
| `expected_qps` | float | Calculated from `user_count` + per-use-case workload parameters | Expected queries per second. Includes peak capacity buffer. Uses `active_fraction` and `requests_per_active_user_per_min` from `usecase_slo_workload.json`. |

#### `QualityWeights` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `categories` | dict[string, int] | From `quality_weights.json` by use case | **Read-only**. Per-category weights for quality scoring. Keys are category names (e.g., `overall`, `coding`, `math`), values are relative integer weights. |

#### `Priorities` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `quality` | `PriorityEntry` | From intent's `quality_priority` | Priority level and resolved weight for model quality |
| `cost` | `PriorityEntry` | From intent's `cost_priority` | Priority level and resolved weight for cost efficiency |
| `latency` | `PriorityEntry` | From intent's `latency_priority` | Priority level and resolved weight for latency/SLO headroom |

Each `PriorityEntry` contains:

```json
{"priority": "low" | "medium" | "high", "weight": int}
```

The `weight` is resolved from the `priority` level using `data/configuration/priority_weights.json`. Both are included so the user can modify either.

### `RankedRecommendations`

The output of `generate-recommendations`. Contains 4 ranked views of deployment configurations.

| Field | Type | Notes |
|---|---|---|
| `specification` | `DeploymentSpecification` | The specification used to generate recommendations (echoed back) |
| `balanced` | list of `DeploymentRecommendation` | Sorted by weighted composite score |
| `best_quality` | list of `DeploymentRecommendation` | Sorted by model capability |
| `lowest_cost` | list of `DeploymentRecommendation` | Sorted by price efficiency |
| `lowest_latency` | list of `DeploymentRecommendation` | Sorted by SLO headroom |
| `total_configs_evaluated` | integer | Total configurations considered |
| `configs_after_filters` | integer | Configurations after SLO/quality/cost filters |

### `DeploymentRecommendation`

A single recommended configuration. Selected from `RankedRecommendations` for user review.

| Field | Type | Notes |
|---|---|---|
| `intent` | `DeploymentIntent` | Original intent |
| `traffic_profile` | `TrafficProfile` | Traffic profile used |
| `slo_targets` | `SLOTargets` | SLO targets used |
| `model_id` | string or null | Recommended model identifier |
| `model_name` | string or null | Human-readable model name |
| `model_uri` | string or null | Model artifact URI |
| `gpu_config` | `GPUConfig` or null | GPU type, count, tensor parallelism |
| `predicted_ttft_p95_ms` | integer or null | Predicted TTFT |
| `predicted_itl_p95_ms` | integer or null | Predicted ITL |
| `predicted_e2e_p95_ms` | integer or null | Predicted E2E latency |
| `predicted_throughput_qps` | float or null | Predicted throughput |
| `benchmark_metrics` | dict or null | All percentile metrics from benchmark |
| `cost_per_hour_usd` | float or null | Hourly cost estimate |
| `cost_per_month_usd` | float or null | Monthly cost estimate |
| `meets_slo` | boolean | Whether configuration meets SLO targets |
| `reasoning` | string | Explanation of recommendation |
| `alternative_options` | list[dict] or null | Alternative configurations with trade-offs |
| `scores` | `ConfigurationScores` or null | Multi-criteria scores for ranking (see below) |
| `configuration` | `DeploymentConfiguration` or null | Embedded deployment parameters for YAML generation. Extract and pass to `generate-deployment`. |

`ConfigurationScores` fields:

| Field | Type | Notes |
|---|---|---|
| `quality_score` | float | Model quality/capability score (0–100) |
| `price_score` | int | Cost efficiency score (0–100) |
| `latency_score` | int | SLO headroom score (0–100) |
| `balanced_score` | float | Weighted composite score (0–100) |
| `slo_status` | string | `"compliant"`, `"near_miss"`, or `"exceeds"` |

**Note on prediction field naming:** The prediction fields (`predicted_ttft_p95_ms`, `predicted_itl_p95_ms`, `predicted_e2e_p95_ms`) keep the `_p95_` infix because they are always p95 predictions from benchmark data. The target fields (`ttft_target_ms`, etc.) drop `_p95_` because the percentile is specified separately via the `percentile` field on `SLOTargets`.

### `DeploymentConfiguration`

A slim model containing only the fields needed for YAML generation. Extracted from `DeploymentRecommendation` and passed to `generate-deployment`.

| Field | Type | Notes |
|---|---|---|
| `model_id` | string | Model identifier (e.g., `meta-llama/Llama-3.1-8B-Instruct`) |
| `model_name` | string | Human-readable model name (e.g., `Llama 3.1 8B Instruct`) |
| `model_uri` | string | Model artifact URI |
| `gpu_config` | `GPUConfig` | GPU type, count, tensor parallelism |
| `use_case` | string | Use case (e.g., `chatbot_conversational`) |
| `expected_qps` | float | Expected queries per second |
| `prompt_tokens` | integer | Mean input token length |
| `output_tokens` | integer | Mean output token length |
| `e2e_target_ms` | integer | End-to-end latency target |

### `DeploymentBundle`

The output of `generate-deployment` and the input to `deploy-bundle-to-cluster`.

| Field | Type | Notes |
|---|---|---|
| `deployment_id` | string | Unique deployment identifier |
| `namespace` | string | Kubernetes namespace |
| `stack` | string | `"vllm"` or `"llm-d"` |
| `configuration` | `DeploymentConfiguration` | The configuration used to generate these files |
| `files` | dict[string, string] | Map of filename to YAML content (e.g., `{"inferenceservice": "...", "autoscaling": "..."}`). Files are applied in iteration order by `deploy-bundle-to-cluster`; order matters if resources have dependencies. |

## API Details

### `POST /api/v1/extract-intent`

Extract structured intent from natural language using an LLM. This is the existing `POST /api/v1/extract` endpoint, renamed for clarity.

**Request:**
```json
{"text": "I need a chatbot for 1000 users, low latency is critical"}
```

**Response:** A `DeploymentIntent` with all fields populated (defaults filled in).

```json
{
  "use_case": "chatbot_conversational",
  "user_count": 1000,
  "domain_specialization": ["general"],
  "preferred_gpu_types": [],
  "preferred_models": [],
  "quality_priority": "medium",
  "cost_priority": "medium",
  "latency_priority": "high"
}
```

**Requires:** LLM (Ollama or configured provider)

### `POST /api/v1/generate-specification`

Generate a complete deployment specification from structured intent. No LLM required.

**Request:** A `DeploymentIntent`. Only `use_case` and `user_count` are required; all other fields are filled with defaults.

**Response:** A `DeploymentSpecification` with all 4 sections populated.

**Requires:** Nothing (reads from static config files)

Example response:
```json
{
  "intent": {
    "use_case": "chatbot_conversational",
    "user_count": 1000,
    "domain_specialization": ["general"],
    "preferred_gpu_types": [],
    "preferred_models": [],
    "quality_priority": "medium",
    "cost_priority": "medium",
    "latency_priority": "high"
  },
  "slo_targets": {
    "ttft_target_ms": 200,
    "itl_target_ms": 24,
    "e2e_target_ms": 6280,
    "percentile": "p95",
    "ttft_range": {"min": 100, "max": 500},
    "itl_range": {"min": 15, "max": 50},
    "e2e_range": {"min": 3940, "max": 13300}
  },
  "workload_profile": {
    "prompt_tokens": 512,
    "output_tokens": 256,
    "expected_qps": 0.87,
  },
  "quality_weights": {
    "categories": {
      "overall": 4,
      "instruction_following": 3,
      "multi_turn": 3,
      "creative_writing": 2,
      "hard_prompts": 2
    }
  },
  "priorities": {
    "quality": {"priority": "medium", "weight": 4},
    "cost": {"priority": "medium", "weight": 4},
    "latency": {"priority": "high", "weight": 2}
  }
}
```

Note: `latency_priority` is `"high"`, so SLO defaults use the 25th percentile of each range (tighter targets).

### `POST /api/v1/generate-recommendations`

Generate ranked deployment recommendations from a specification.

**Request:** A `RecommendationRequest` containing a `DeploymentSpecification` (the output of `generate-specification`, with optional user modifications). The weights come from the specification's `priorities` section.

Additional optional fields on the request (not part of the specification):
- `enable_estimated: bool = true` — run roofline estimation for missing benchmarks
- `min_quality: float | null` — minimum quality score filter
- `max_cost: float | null` — maximum cost filter
- `include_near_miss: bool = true` — include near-miss configurations

```json
{
  "specification": {
    "intent": {...},
    "slo_targets": {...},
    "workload_profile": {...},
    "quality_weights": {...},
    "priorities": {...}
  },
  "enable_estimated": true,
  "min_quality": 50.0,
  "max_cost": 10000.0,
  "include_near_miss": true
}
```

**Response:** A `RankedRecommendations` object with 4 ranked views.

```json
{
  "specification": {...},
  "balanced": [
    {
      "model_id": "meta-llama/Llama-3.1-8B-Instruct",
      "model_name": "Llama 3.1 8B Instruct",
      "gpu_config": {
        "gpu_type": "NVIDIA-L4",
        "gpu_count": 1,
        "tensor_parallel": 1,
        "replicas": 1
      },
      "cost_per_month_usd": 350.0,
      "scores": {
        "quality_score": 75.5,
        "price_score": 85,
        "latency_score": 90,
        "balanced_score": 82.3,
        "slo_status": "compliant"
      },
      "..."
    }
  ],
  "best_quality": [...],
  "lowest_cost": [...],
  "lowest_latency": [...]
}
```

**Requires:** Benchmark database

### `POST /api/v1/generate-deployment`

Generate deployment files (YAML) for a selected configuration.

**Request:**
```json
{
  "configuration": {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "model_name": "Llama 3.1 8B Instruct",
    "model_uri": "meta-llama/Llama-3.1-8B-Instruct",
    "gpu_config": {
      "gpu_type": "NVIDIA-L4",
      "gpu_count": 1,
      "tensor_parallel": 1,
      "replicas": 1
    },
    "use_case": "chatbot_conversational",
    "expected_qps": 0.87,
    "prompt_tokens": 512,
    "output_tokens": 256,
    "e2e_target_ms": 6280
  },
  "namespace": "default",
  "stack": "vllm"
}
```

Parameters:
- `configuration` — the `DeploymentConfiguration` extracted from the selected recommendation
- `namespace` — Kubernetes namespace (default: `"default"`)
- `stack` — deployment stack: `"vllm"` or `"llm-d"` (default: `"vllm"`)

The `stack` parameter selects which set of Jinja2 templates to use:
- `vllm` — generates KServe `InferenceService` with vLLM runtime, HPA autoscaling
- `llm-d` — generates llm-d-specific resources (EPP, PD split, routing)

**Response:** A `DeploymentBundle` with generated YAML files.

```json
{
  "deployment_id": "chatbot-1000-users-abc123",
  "namespace": "default",
  "stack": "vllm",
  "configuration": {...},
  "files": {
    "inferenceservice": "apiVersion: serving.kserve.io/v1beta1\nkind: InferenceService\n...",
    "autoscaling": "apiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler\n..."
  }
}
```

**Requires:** Nothing (Jinja2 template rendering)

### `POST /api/v1/deploy-bundle-to-cluster`

Deploy a deployment bundle to a Kubernetes cluster. This endpoint accepts a `DeploymentBundle` (from `generate-deployment`, possibly with user-modified YAML) and deploys it directly without re-generating. YAML content from `bundle.files` is applied directly via `kubectl apply -f -` (piped to stdin) — no intermediate files are written to disk.

**Request:**
```json
{
  "deployment_id": "chatbot-1000-users-abc123",
  "namespace": "default",
  "stack": "vllm",
  "configuration": {...},
  "files": {
    "inferenceservice": "apiVersion: serving.kserve.io/v1beta1\n...",
    "autoscaling": "apiVersion: autoscaling/v2\n..."
  }
}
```

**Requires:** Kubernetes cluster access

## Existing endpoints: what stays, what changes

### Endpoints in the new pipeline

| Current endpoint | New endpoint | Change |
|---|---|---|
| `POST /extract` | `POST /extract-intent` | Rename. Keep `/extract` as alias. Remove `experience_class` from response. |
| *(new)* | `POST /generate-specification` | New endpoint. |
| *(new)* | `POST /generate-recommendations` | New endpoint. Accept structured `DeploymentSpecification` instead of flat fields. |
| `POST /deploy` | `POST /generate-deployment` | Removed. Replaced by `/generate-deployment`. |
| *(new)* | `POST /deploy-bundle-to-cluster` | New endpoint accepting `DeploymentBundle`. |

### Endpoints that stay as-is (UI support)

These serve the Streamlit UI's interactive controls and configuration panels. They remain unchanged:

| Endpoint | Purpose |
|---|---|
| `GET /slo-defaults/{use_case}` | SLO ranges for UI |
| `GET /workload-profile/{use_case}` | Workload display for UI |
| `GET /expected-rps/{use_case}` | QPS estimate for UI |
| `GET /models` | Model catalog for UI dropdowns |
| `GET /gpu-types` | GPU catalog for UI dropdowns |
| `GET /use-cases` | Use case list for UI dropdowns |
| `GET /priority-weights` | Priority weight config for UI |
| `GET /deployment-mode` | Simulator vs production toggle |
| `PUT /deployment-mode` | Simulator vs production toggle |
| `GET /cluster-status` | Cluster monitoring UI |
| `GET /deployments` | Deployment list UI |
| `GET /deployments/{id}/k8s-status` | Deployment status UI |
| `DELETE /deployments/{id}` | Deployment management UI |

### GUI migration to new pipeline endpoints

The GUI has been migrated to use the new pipeline endpoints. The following legacy endpoints may be removed in a future version:

| Legacy endpoint | Replaced by | Status |
|---|---|---|
| `GET /slo-defaults/{use_case}` | `POST /generate-specification` | Can be removed |
| `GET /workload-profile/{use_case}` | `POST /generate-specification` | Can be removed |
| `GET /expected-rps/{use_case}` | `POST /generate-specification` | Can be removed |
| `GET /priority-weights` | `POST /generate-specification` | Can be removed |
| `POST /extract` | `POST /extract-intent` | Alias kept for compatibility |

### Endpoints to remove

| Endpoint | Status | Notes |
|---|---|---|
| `POST /recommend` | **Removed** | One-shot endpoint (extract + recommend + generate YAML). Not called by the GUI. The new pipeline made it redundant. |
| `POST /test` | **Removed** | Quick test endpoint, not needed with proper tests. Not called by the GUI. |
| `GET /deployments/{id}/status` | **Removed** | Returned mock observability data (random numbers). Not called by the GUI (it uses `/k8s-status`). Real metrics are future work. |
| `POST /ranked-recommend-from-spec` | **Removed** | Replaced by `POST /generate-recommendations`. |

### Endpoints that stay

| Endpoint | Notes |
|---|---|
| `GET /models` | Model catalog for UI dropdowns — not redundant with pipeline |
| `GET /gpu-types` | GPU catalog for UI dropdowns — not redundant with pipeline |
| `GET /use-cases` | Use case list for UI dropdowns — not redundant with pipeline |
| `POST /model-info` | Capacity Planner page — separate concern from pipeline |
| `POST /calculate` | Capacity Planner page — separate concern from pipeline |
| `POST /estimate` | Capacity Planner page — separate concern from pipeline |
| `GET /deployment-mode`, `PUT /deployment-mode` | Simulator vs production toggle |
| `GET /cluster-status` | Cluster monitoring |
| `GET /deployments`, `DELETE /deployments/{id}` | Deployment management |
| `GET /deployments/{id}/k8s-status` | Deployment status |
| `GET /db/*`, `POST /db/*` | Database management |
| `GET /quality/*`, `PUT /quality/*`, `POST /quality/*` | Quality data management |

### Backend functions to evaluate

During implementation, evaluate each backend function that currently serves the old endpoints:

- Functions behind `/slo-defaults`, `/workload-profile`, `/expected-rps`, `/priority-weights`: their logic should be folded into the `generate-specification` implementation. Remove the standalone functions if no other callers exist.
- `RecommendationWorkflow.generate_specification()`: bundles LLM extraction with specification generation. The new `generate-specification` endpoint calls `TrafficProfileGenerator` directly. Evaluate whether the workflow method is still needed.
- `POST /recommend` handler and its orchestration logic: remove entirely.

## GUI changes

- **Intent form:** Added an option to skip LLM intent extraction and fill out the `DeploymentIntent` as a form directly (Form mode). This supports users without an LLM available.

- **SLO input:** Fixed regression where the GUI restricted SLO values to the recommended range. The range is shown as a recommendation, but users can enter any positive value. Uses the latency-priority-influenced default percentile (high=25th, medium=50th, low=75th) for initial values, matching the API behavior.

- **Technical Specification tab:** Remains as-is — these are the same 4 sections the API returns. The user can edit them before generating recommendations.

## Implementation Status

All planned changes have been implemented:

### Completed Changes

1. ✅ **Removed `experience_class`** from schemas, intent extraction, workflow orchestration, and SLO templates
2. ✅ **Added GPU count limits** via `GpuPreference` model in `DeploymentIntent`
3. ✅ **Defined named objects** (`SLOTargets`, `WorkloadProfile`, `QualityWeights`, `Priorities`, `DeploymentSpecification`, `DeploymentBundle`)
4. ✅ **Updated SLO default generation** to use range-percentile-based defaults (high=25th, medium=50th, low=75th)
5. ✅ **Added `POST /extract-intent`** (kept `/extract` as alias)
6. ✅ **Added `POST /generate-specification`** endpoint
7. ✅ **Added `POST /generate-recommendations`** endpoint (removed `/ranked-recommend-from-spec`)
8. ✅ **Added `POST /generate-deployment`** endpoint (removed `/deploy`)
9. ✅ **Added `POST /deploy-bundle-to-cluster`** endpoint
10. ✅ **Added comprehensive tests** for all new endpoints with deterministic fixture data
11. ✅ **Migrated GUI** to use new pipeline endpoints with Form mode (skip-LLM) option

## Files Modified

All planned files were successfully modified during implementation:

| File | Change |
|---|---|
| `src/planner/shared/schemas/intent.py` | ✅ Removed `experience_class`; added `GpuPreference` model; updated `preferred_gpu_types` type |
| `src/planner/shared/schemas/specification.py` | ✅ Defined named objects (`SLOTargets`, `WorkloadProfile`, `QualityWeights`, `Priorities`, `DeploymentSpecification`) |
| `src/planner/shared/schemas/recommendation.py` | ✅ Added `DeploymentBundle`; renamed `RankedRecommendationsResponse` to `RankedRecommendations` |
| `src/planner/intent_extraction/extractor.py` | ✅ Removed `experience_class` inference from `infer_missing_fields()` |
| `src/planner/orchestration/workflow.py` | ✅ Removed `experience_class` inference blocks and log reference |
| `src/planner/knowledge_base/slo_templates.py` | ✅ Removed `experience_class` from `SLOTemplate` and `get_templates_by_experience_class()` |
| `src/planner/shared/utils/gpu_normalizer.py` | ✅ Handle `GpuPreference` objects in `normalize_gpu_types()` |
| `src/planner/recommendation/config_finder.py` | ✅ Respect `max_count` GPU limits when filtering configurations |
| `src/planner/specification/traffic_profile.py` | ✅ Replaced multiplier-based SLO adjustment with range-percentile defaults |
| `src/planner/api/routes/intent.py` | ✅ Renamed `POST /extract` to `POST /extract-intent` (kept alias) |
| `src/planner/api/routes/specification.py` | ✅ Added `POST /generate-specification` endpoint |
| `src/planner/api/routes/recommendation.py` | ✅ Added `POST /generate-recommendations`; removed `/ranked-recommend-from-spec` |
| `src/planner/api/routes/configuration.py` | ✅ Added `POST /generate-deployment` and `POST /deploy-bundle-to-cluster` |
| `ui/components/slo.py` | ✅ Use range-percentile defaults; removed min/max enforcement on SLO inputs |
| `tests/` | ✅ Added comprehensive test files for pipeline endpoints |
| `docs/ARCHITECTURE.md` | To be updated (this branch) |

## Notes

- **`experience_class` removal**: This field is a 1:1 deterministic mapping from `use_case` and is not consumed by any downstream engine. Remove it from the schema, intent extraction, workflow orchestration, and SLO templates. If it's ever needed in the future, derive it on the fly from `use_case` at the point of use.
- **`domain_specialization`**: Not currently used by the recommendation engine, but kept on the intent schema. Future work: use it to influence quality benchmark weight selection (e.g., a `code` domain could boost coding-related category weights).
- **GPU count limits**: The `max_count` field is a new concept. It needs to flow through GPU normalization (which currently works with plain strings) and into `ConfigFinder.find_optimal_configurations()`. Configurations where `tensor_parallel > max_count` for a given GPU type should be filtered out.
- **SLO percentile**: The `percentile` field (p90/p95/p99) is on the schema now but only p95 is supported by the backend benchmark data. The endpoint should validate and return an error for unsupported values until backend support is added.
- **Backward compatibility**: Old endpoint path `/extract` is kept as an alias for `/extract-intent`. `/deploy`, `/deploy-to-cluster`, and `/ranked-recommend-from-spec` have been fully removed.
- **Express path**: A future enhancement could add a `through` parameter (e.g., `POST /generate-specification?through=recommendations`) to skip intermediate steps. Not designed here.
- The quality weights and priority weights are currently read from JSON files by the UI components. The new endpoints read the same files, keeping them as the single source of truth.
- The existing `RecommendationWorkflow.generate_specification()` method bundles LLM extraction with specification generation. The new `generate-specification` endpoint bypasses it and calls `TrafficProfileGenerator` directly, which is the cleaner separation.

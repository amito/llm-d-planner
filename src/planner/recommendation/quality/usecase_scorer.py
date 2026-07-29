"""Use-case specific model quality scoring based on Artificial Analysis benchmarks.

This module provides quality/accuracy scores for models based on their performance
on task-specific benchmarks (MMLU-Pro, LiveCodeBench, IFBench, etc.).

Integration with Planner:
- Provides use-case specific quality scores (replaces legacy size-based heuristics)
- Andre's latency/throughput benchmarks from PostgreSQL are KEPT as-is
- The final recommendation combines: Our quality + Andre's latency/cost scoring
"""

import csv
import logging
import os
import re

from planner.data._resolver import data_path

logger = logging.getLogger(__name__)

_SIZE_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?[bB])\b")
_QUANT_TOKENS = frozenset({"fp8", "dynamic", "nvfp4", "hf"})
_DATE_SUFFIX_RE = re.compile(r"^2[0-9]{3}$")

WEIGHTED_SCORES_DIR = str(data_path("benchmarks/accuracy/weighted_scores"))


class UseCaseQualityScorer:
    """Score models based on use-case specific benchmark performance.

    Uses pre-calculated weighted scores from Artificial Analysis benchmarks.
    Each use case has different weights for different benchmarks:
    - chatbot_conversational: MMLU-Pro (30%), IFBench (30%), HLE (20%), etc.
    - code_completion: LiveCodeBench (35%), SciCode (30%), etc.
    - See USE_CASE_METHODOLOGY.md for full details.
    """

    # Mapping from use case to CSV filename
    USE_CASE_FILES = {
        "chatbot_conversational": "opensource_chatbot_conversational.csv",
        "code_completion": "opensource_code_completion.csv",
        "code_generation_detailed": "opensource_code_generation_detailed.csv",
        "translation": "opensource_translation.csv",
        "content_generation": "opensource_content_generation.csv",
        "summarization_short": "opensource_summarization_short.csv",
        "document_analysis_rag": "opensource_document_analysis_rag.csv",
        "long_document_summarization": "opensource_long_document_summarization.csv",
        "research_legal_analysis": "opensource_research_legal_analysis.csv",
    }

    def __init__(self):
        """Initialize the scorer with cached data."""
        self._cache: dict[str, dict[str, float]] = {}
        self._token_index: dict[str, dict[frozenset[str], str]] = {}
        self._catalog_fallback: dict[str, float] = {}
        self._load_all_scores()

    def _load_all_scores(self):
        """Pre-load all use case scores into memory."""
        for use_case, filename in self.USE_CASE_FILES.items():
            filepath = os.path.join(WEIGHTED_SCORES_DIR, filename)
            if os.path.exists(filepath):
                self._cache[use_case] = self._load_csv_scores(filepath)
                token_map: dict[frozenset[str], str] = {}
                for model_name in self._cache[use_case]:
                    tokens = self._tokenize_for_matching(model_name)
                    if tokens not in token_map:
                        token_map[tokens] = model_name
                self._token_index[use_case] = token_map
                logger.info(f"Loaded {len(self._cache[use_case])} model scores for {use_case}")
            else:
                logger.warning(f"Weighted scores file not found: {filepath}")
                self._cache[use_case] = {}

    def _load_csv_scores(self, filepath: str) -> dict[str, float]:
        """Load scores from a weighted_scores CSV file.

        Returns:
            Dict mapping model name (lowercase) to score (0-100)
        """
        scores = {}
        try:
            with open(filepath, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Support both old format (Model Name, Use Case Score) and new format (model_name, weighted_score)
                    model_name = row.get("model_name", row.get("Model Name", "")).strip()
                    score_str = row.get("weighted_score", row.get("Use Case Score", "0"))

                    # Parse score (handle percentage strings)
                    try:
                        if "%" in str(score_str):
                            score = float(score_str.replace("%", ""))
                        else:
                            score = (
                                float(score_str) * 100
                                if float(score_str) <= 1
                                else float(score_str)
                            )
                    except (ValueError, TypeError):
                        score = 0.0

                    if model_name:
                        # Store with lowercase key for easier matching
                        scores[model_name.lower()] = min(100, max(0, score))
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")

        return scores

    def set_catalog_fallback(self, scores: dict[str, float]) -> None:
        """Set catalog-sourced quality scores as a fallback.

        These scores are only used when a model has no CSV match.
        Values are normalized to the 0-100 scale (inputs <= 1.0 are
        treated as fractions and scaled up).
        """
        normalized: dict[str, float] = {}
        for k, v in scores.items():
            score = v * 100.0 if 0 < v <= 1.0 else v
            normalized[k.lower()] = min(100.0, max(0.0, score))
        self._catalog_fallback = normalized
        logger.info("Set catalog fallback scores for %d models", len(self._catalog_fallback))

    # Map from normalized model name -> AA weighted score CSV name.
    # Keys must match output of _normalize_model_name() (org prefix stripped,
    # quantization/date suffixes removed). One entry per base model; all
    # quantized/FP8 variants collapse to the same normalized key.
    BENCHMARK_TO_AA_MAP = {
        "gpt-oss-120b": "gpt-oss-120b (high)",
        "gpt-oss-20b": "gpt-oss-20b (high)",
        "deepseek-r1-0528": "deepseek r1 0528 (may '25)",
        "kimi-k2": "kimi k2",
        "llama-4-maverick-17b-128e": "llama 4 maverick",
        "qwen2.5-7b": "qwen2.5 max",
        "llama-3.3-70b": "llama 3.3 instruct 70b",
        "llama-4-scout-17b-16e": "llama 4 scout",
        "qwen3-8b": "qwen3 8b (reasoning)",
        "nvidia-nemotron-nano-9b-v2": "nvidia nemotron nano 9b v2 (reasoning)",
        "qwen3-coder-480b-a35b": "qwen3 coder 480b a35b instruct",
        "llama-3.1-nemotron-70b": "llama 3.1 nemotron instruct 70b",
        "mistral-small-3.1-24b": "mistral small 3.1",
        "phi-4": "phi-4",
        "mistral-small-24b": "mistral small 3",
        "llama-3.1-8b": "llama 3.1 instruct 8b",
        "meta-llama-3.1-8b": "llama 3.1 instruct 8b",
        "gemma-3n-e4b-it": "gemma 3n e4b instruct",
        "granite-3.1-8b": "granite 3.3 8b (non-reasoning)",
        "mixtral-8x7b": "mixtral 8x7b instruct",
        "ministral-3-14b": "ministral 8b",  # no 14B in AA; 8B is only Ministral variant
        "qwen3-next-80b-a3b": "qwen3 next 80b a3b (reasoning)",
        "qwen3-vl-235b-a22b": "qwen3 vl 235b a22b (reasoning)",
        "granite-4.0-h-small": "granite 4.0 h small",
        "granite-4.0-h-tiny": "granite 4.0 h 1b",  # catalog lists Tiny as 1B; H 1B is closest
        "mistral-large-3-675b": "mistral large 2 (nov '24)",
        "nvidia-nemotron-3-nano-30b-a3b": "llama 3.1 nemotron nano 4b v1.1 (reasoning)",  # MoE with 3B active; 4B is closest
    }

    # Quantization discount factors applied to base model quality scores.
    # FP8 has negligible quality loss; W8A8 is minor; W4A16/NVFP4 lose more.
    QUANTIZATION_DISCOUNTS: dict[str, float] = {
        "-fp8-dynamic": 1.0,
        "-fp8": 1.0,
        "-quantized.w8a8": 0.97,
        "-quantized.w4a16": 0.92,
        "-nvfp4": 0.92,
    }

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name by removing quantization suffixes and org prefixes."""
        name = model_name.lower()

        # Remove org prefixes
        if "/" in name:
            name = name.split("/")[-1]

        # Longer suffixes must appear before shorter ones (e.g., -fp8-dynamic before -fp8)
        suffixes_to_remove = [
            "-nvfp4",
            "-fp8-dynamic",
            "-fp8",
            "-quantized.w4a16",
            "-quantized.w8a8",
            "-instruct-2501",
            "-instruct-2503",
            "-instruct-2509",
            "-instruct-2512",
            "-instruct-hf",
            "-instruct-v0.1",
            "-instruct",
            "-reasoning",
        ]
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[: -len(suffix)]

        return name.strip("-").strip()

    def _tokenize_for_matching(self, name: str) -> frozenset[str]:
        """Tokenize model name into a word set for order-independent matching."""
        name = name.lower()
        if "/" in name:
            name = name.split("/")[-1]
        name = re.sub(r"[()'\",]", " ", name)
        tokens = re.split(r"[-_\s]+", name)
        return frozenset(
            t
            for t in tokens
            if t
            and t not in _QUANT_TOKENS
            and not t.startswith("quantized.")
            and not _DATE_SUFFIX_RE.match(t)
        )

    def _get_quantization_discount(self, model_name: str) -> float:
        """Return a discount factor (0-1) based on the quantization level."""
        name = model_name.lower()
        if "/" in name:
            name = name.split("/")[-1]
        for suffix, factor in self.QUANTIZATION_DISCOUNTS.items():
            if suffix in name:
                return factor
        return 1.0

    def get_quality_score(self, model_name: str, use_case: str) -> float:
        """Get quality score for a model on a specific use case.

        Args:
            model_name: Model name (e.g., "Llama 3.1 Instruct 8B", "meta-llama/llama-3.1-8b-instruct")
            use_case: Use case identifier (e.g., "code_completion")

        Returns:
            Quality score 0-100 (higher is better), or 0 if no valid AA data
        """
        # Normalize use case
        use_case_normalized = use_case.lower().replace(" ", "_").replace("-", "_")

        if use_case_normalized not in self._cache:
            logger.warning(f"Unknown use case: {use_case}, using chatbot_conversational")
            use_case_normalized = "chatbot_conversational"

        scores = self._cache.get(use_case_normalized, {})

        # Normalize the model name
        model_lower = model_name.lower()
        base_model = self._normalize_model_name(model_name)
        discount = self._get_quantization_discount(model_name)

        # Try exact match first
        if model_lower in scores:
            return scores[model_lower]

        # Try order-independent word-set match (same words, different order)
        input_tokens = self._tokenize_for_matching(model_name)
        token_map = self._token_index.get(use_case_normalized, {})
        token_match = token_map.get(input_tokens)
        if token_match is not None and token_match in scores:
            logger.debug(f"Token-set match {model_name} -> {token_match}")
            return round(scores[token_match] * discount, 2)

        # Try benchmark to AA mapping (for known valid models)
        aa_name = self.BENCHMARK_TO_AA_MAP.get(base_model)
        if aa_name is not None and aa_name in scores:
            logger.debug(f"Matched {model_name} -> {aa_name} via benchmark mapping")
            return round(scores[aa_name] * discount, 2)

        # Try partial matching (for HuggingFace repo names)
        # Find BEST match - prioritize matches that include model size (7b, 20b, 120b, etc.)
        model_sizes = {s.lower() for s in _SIZE_PATTERN.findall(base_model)}

        best_match = None
        best_score = 0.0
        best_common_count = 0
        best_has_size_match = False

        for cached_name, score in scores.items():
            model_words = set(
                base_model.replace("-", " ").replace("/", " ").replace("_", " ").split()
            )
            cached_words = set(
                cached_name.replace("-", " ").replace("/", " ").replace("_", " ").split()
            )

            common_words = model_words & cached_words
            if len(common_words) >= 2:
                # Check if this match includes the model size
                cached_sizes = {s.lower() for s in _SIZE_PATTERN.findall(cached_name)}
                has_size_match = bool(model_sizes & cached_sizes)

                # Prefer matches with size match, then by common word count
                is_better = False
                if has_size_match and not best_has_size_match:
                    is_better = True  # Size match beats no size match
                elif (
                    has_size_match == best_has_size_match and len(common_words) > best_common_count
                ):
                    is_better = True  # More common words is better

                if is_better:
                    best_match = cached_name
                    best_score = score
                    best_common_count = len(common_words)
                    best_has_size_match = has_size_match

        if best_match:
            logger.debug(
                f"Partial match {model_name} -> {best_match} (size_match={best_has_size_match})"
            )
            return round(best_score * discount, 2)

        # Check catalog fallback (try full name, then base model)
        if self._catalog_fallback:
            fallback = self._catalog_fallback.get(model_lower, 0.0)
            if fallback <= 0:
                fallback = self._catalog_fallback.get(base_model, 0.0)
            if fallback > 0:
                logger.debug(f"Using catalog fallback score for {model_name}: {fallback}")
                return fallback

        # No valid AA data found - return 0 to indicate missing data
        # This allows filtering out models without quality scores
        logger.debug(f"No AA score found for {model_name} (base: {base_model})")
        return 0.0  # Return 0 so min_accuracy filter can exclude these

    def get_top_models_for_usecase(self, use_case: str, top_n: int = 10) -> list[tuple[str, float]]:
        """Get top N models for a specific use case."""
        use_case_normalized = use_case.lower().replace(" ", "_").replace("-", "_")
        scores = self._cache.get(use_case_normalized, {})
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_models[:top_n]

    def get_available_use_cases(self) -> list[str]:
        """Get list of available use cases."""
        return list(self.USE_CASE_FILES.keys())


# Singleton instance
_scorer_instance: UseCaseQualityScorer | None = None


def get_quality_scorer() -> UseCaseQualityScorer:
    """Get the singleton quality scorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = UseCaseQualityScorer()
    return _scorer_instance


def score_model_quality(model_name: str, use_case: str) -> float:
    """Convenience function to get quality score."""
    return get_quality_scorer().get_quality_score(model_name, use_case)

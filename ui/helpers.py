"""Pure utility functions for the Planner UI.

No Streamlit dependency — safe to import anywhere.
"""


def format_use_case_name(use_case: str) -> str:
    """Format use case name with proper capitalization for acronyms."""
    if not use_case:
        return "Unknown"
    # Replace underscores and title case
    formatted = use_case.replace("_", " ").title()
    # Fix common acronyms
    acronyms = {
        "Rag": "RAG",
        "Llm": "LLM",
        "Ai": "AI",
        "Api": "API",
        "Gpu": "GPU",
        "Cpu": "CPU",
        "Slo": "SLO",
        "Qps": "QPS",
        "Rps": "RPS",
    }
    for wrong, right in acronyms.items():
        formatted = formatted.replace(wrong, right)
    return formatted


def get_scores(rec: dict) -> dict:
    """Extract normalized scores from a backend recommendation."""
    backend_scores = rec.get("scores", {}) or {}
    return {
        "quality": backend_scores.get("quality_score", 0),
        "latency": backend_scores.get("latency_score", 0),
        "cost": backend_scores.get("price_score", 0),
        "final": backend_scores.get("balanced_score", 0),
    }


def format_gpu_config(gpu_config: dict) -> str:
    """Format GPU configuration for display.

    Example: "2x A100 (TP=2, R=1)"
    """
    if not isinstance(gpu_config, dict):
        return "Unknown"
    gpu_type = gpu_config.get("gpu_type", "Unknown")
    gpu_count = gpu_config.get("gpu_count", 1)
    tp = gpu_config.get("tensor_parallel", 1)
    replicas = gpu_config.get("replicas", 1)
    return f"{gpu_count}x {gpu_type} (TP={tp}, R={replicas})"

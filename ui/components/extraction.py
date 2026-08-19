"""Extraction display components for the Planner UI.

Extraction result display, approval workflow, and edit form.
"""

import streamlit as st
from api_client import generate_specification

from components.shared_intent_form import render_intent_fields


def _format_priorities(extraction: dict) -> str:
    """Format priority display from extraction data."""
    quality = extraction.get("quality_priority", "medium")
    cost = extraction.get("cost_priority", "medium")
    latency = extraction.get("latency_priority", "medium")

    parts = []
    if quality != "medium":
        parts.append(f"Quality: {quality.title()}")
    if cost != "medium":
        parts.append(f"Cost: {cost.title()}")
    if latency != "medium":
        parts.append(f"Latency: {latency.title()}")

    return ", ".join(parts) if parts else "Default"


def _format_models(extraction: dict) -> str:
    """Format preferred models display."""
    models = extraction.get("preferred_models", []) or st.session_state.get("preferred_models", [])
    if not models:
        return "Any"
    return ", ".join(models)


def render_extraction_result(extraction: dict, priority: str):
    """Render extraction results (read-only, after approval) with modify button."""
    st.subheader("Extracted Business Context")

    use_case = extraction.get("use_case", "unknown")
    use_case_display = use_case.replace("_", " ").title() if use_case else "Unknown"
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware") or "Any GPU"
    priorities = _format_priorities(extraction)
    models = _format_models(extraction)

    st.markdown(
        f"**Use Case:** {use_case_display}  \n"
        f"**Expected Users:** {user_count:,}  \n"
        f"**Hardware:** {hardware}  \n"
        f"**Models:** {models}  \n"
        f"**Priorities:** {priorities}"
    )

    if st.button("Modify Business Context", use_container_width=False, key="modify_after_approve"):
        st.session_state.extraction_approved = False
        st.session_state.slo_approved = None
        st.session_state.recommendation_result = None
        st.rerun()


def render_extraction_with_approval(extraction: dict):
    """Render extraction results with YES/NO approval buttons."""
    st.subheader("Extracted Business Context")

    use_case = extraction.get("use_case", "unknown")
    use_case_display = use_case.replace("_", " ").title() if use_case else "Unknown"
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware") or "Any GPU"
    priorities = _format_priorities(extraction)
    models = _format_models(extraction)

    st.markdown(
        f"**Use Case:** {use_case_display}  \n"
        f"**Expected Users:** {user_count:,}  \n"
        f"**Hardware:** {hardware}  \n"
        f"**Models:** {models}  \n"
        f"**Priorities:** {priorities}"
    )

    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    with col1:
        if st.button(
            "Generate Specification",
            type="primary",
            width="stretch",
            key="approve_extraction",
        ):
            st.session_state.extraction_approved = True
            st.session_state._pending_tab = 1
            st.rerun()
    with col2:
        if st.button("Modify Extracted Context", width="stretch", key="edit_extraction"):
            st.session_state.extraction_approved = False
            st.rerun()
    with col3:
        if st.button("Start Over", width="stretch", key="restart"):
            st.session_state.extraction_result = None
            st.session_state.extraction_approved = None
            st.session_state.recommendation_result = None
            st.session_state.user_input = ""
            for key in [
                "quality_priority",
                "cost_priority",
                "latency_priority",
                "weight_quality",
                "weight_cost",
                "weight_latency",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def render_extraction_edit_form(extraction: dict):
    """Render editable form for extraction correction using shared intent fields."""
    st.subheader("Edit Business Context")
    st.info(
        'Review and adjust the extracted values below, then click "Apply Changes" '
        "to regenerate the specification."
    )

    intent = render_intent_fields(defaults=extraction, key_prefix="edit")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        if st.button("Apply Changes", type="primary", width="stretch", key="apply_edit"):
            with st.spinner("Regenerating specification..."):
                specification = generate_specification(intent)

            if specification:
                edited = {
                    "use_case": specification["intent"]["use_case"],
                    "user_count": specification["intent"]["user_count"],
                    "quality_priority": specification["intent"]["quality_priority"],
                    "cost_priority": specification["intent"]["cost_priority"],
                    "latency_priority": specification["intent"]["latency_priority"],
                    "preferred_gpu_types": specification["intent"]["preferred_gpu_types"],
                    "preferred_models": specification["intent"]["preferred_models"],
                }

                # Store SLO targets
                slo = specification["slo_targets"]
                st.session_state.input_ttft = slo["ttft_target_ms"]
                st.session_state.custom_ttft = slo["ttft_target_ms"]
                st.session_state.input_itl = slo["itl_target_ms"]
                st.session_state.custom_itl = slo["itl_target_ms"]
                st.session_state.input_e2e = slo["e2e_target_ms"]
                st.session_state.custom_e2e = slo["e2e_target_ms"]
                st.session_state.slo_percentile = slo["percentile"]

                # Store workload profile
                workload = specification["workload_profile"]
                st.session_state.spec_prompt_tokens = workload["prompt_tokens"]
                st.session_state.spec_output_tokens = workload["output_tokens"]
                st.session_state.spec_expected_qps = workload["expected_qps"]

                # Store priorities
                priorities = specification["priorities"]
                st.session_state.weight_quality = priorities["quality"]["weight"]
                st.session_state.weight_cost = priorities["cost"]["weight"]
                st.session_state.weight_latency = priorities["latency"]["weight"]
                st.session_state.quality_priority = specification["intent"]["quality_priority"]
                st.session_state.cost_priority = specification["intent"]["cost_priority"]
                st.session_state.latency_priority = specification["intent"]["latency_priority"]

                st.session_state.preferred_models = edited["preferred_models"]
                st.session_state.edited_extraction = edited
                st.session_state.extraction_result.update(edited)
                st.session_state.extraction_approved = True
                st.session_state.slo_approved = None
                st.session_state.recommendation_result = None
                st.session_state.pop("_last_spec_fingerprint", None)
                st.session_state.pop("_specification_populated", None)

                st.session_state._pending_tab = 1
                st.rerun()
            else:
                st.error("Failed to regenerate specification. Check backend logs.")
    with col2:
        if st.button("Cancel", width="stretch", key="cancel_edit"):
            st.session_state.extraction_approved = None
            st.rerun()

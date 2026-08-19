"""Intent form component for manual DeploymentIntent specification.

Allows users to fill out DeploymentIntent fields directly without LLM extraction.
"""

import logging

import streamlit as st
from api_client import generate_specification

from components.shared_intent_form import render_intent_fields

logger = logging.getLogger(__name__)


def render_intent_form():
    """Render form to manually specify DeploymentIntent without LLM extraction."""
    st.subheader("Manual Intent Specification")
    st.caption(
        "Fill out the deployment requirements directly without using the LLM extraction. "
        "This mode is useful when you know exactly what you need."
    )

    intent = render_intent_fields(key_prefix="form")

    # Submit button
    if st.button("Generate Specification", type="primary", width="stretch"):
        with st.spinner("Generating specification from intent..."):
            specification = generate_specification(intent)

        if specification:
            # Populate session state with the specification data
            st.session_state.extraction_result = {
                "use_case": specification["intent"]["use_case"],
                "user_count": specification["intent"]["user_count"],
                "quality_priority": specification["intent"]["quality_priority"],
                "cost_priority": specification["intent"]["cost_priority"],
                "latency_priority": specification["intent"]["latency_priority"],
                "preferred_gpu_types": specification["intent"]["preferred_gpu_types"],
                "preferred_models": specification["intent"]["preferred_models"],
            }

            # Store SLO targets
            slo_targets = specification["slo_targets"]
            st.session_state.input_ttft = slo_targets["ttft_target_ms"]
            st.session_state.custom_ttft = slo_targets["ttft_target_ms"]
            st.session_state.input_itl = slo_targets["itl_target_ms"]
            st.session_state.custom_itl = slo_targets["itl_target_ms"]
            st.session_state.input_e2e = slo_targets["e2e_target_ms"]
            st.session_state.custom_e2e = slo_targets["e2e_target_ms"]
            st.session_state.slo_percentile = slo_targets["percentile"]

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

            # Mark as approved to skip LLM extraction step
            st.session_state.extraction_approved = True
            st.session_state.slo_approved = None
            st.session_state.recommendation_result = None
            st.session_state.preferred_models = specification["intent"]["preferred_models"]

            # Clear previous recommendation selection and deployment state
            st.session_state.deployment_selected_config = None
            st.session_state.deployment_selected_category = None
            st.session_state.deployment_yaml_generated = False
            st.session_state.deployment_yaml_files = {}
            st.session_state.deployment_id = None
            st.session_state.deployment_error = None
            st.session_state.pop("_last_spec_fingerprint", None)
            st.session_state.pop("_specification_populated", None)

            st.session_state._pending_tab = 1
            st.rerun()
        else:
            st.error("Failed to generate specification. Please check backend logs.")

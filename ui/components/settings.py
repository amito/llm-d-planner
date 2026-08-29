"""Settings / Configuration tab component.

Contains benchmark database management controls;
structured to support additional configuration sections.
"""

import requests
import streamlit as st
from api_client import (
    API_BASE_URL,
    fetch_db_status,
    fetch_deployment_mode,
    is_db_admin_required,
    reset_database,
    update_deployment_mode,
    upload_benchmarks,
    verify_db_admin_password,
)

_TAB_INDEX = 5  # Configuration is the 6th tab (0-indexed)


def render_configuration_tab():
    """Render the Configuration tab with deployment mode and database management."""
    # --- Deployment Mode ---
    st.subheader("Deployment Mode")

    # Sync deployment mode from backend on each render
    current_mode = fetch_deployment_mode()
    st.session_state.deployment_mode_selection = (
        "Simulator" if current_mode == "simulator" else "Production"
    )

    def _on_mode_change():
        new_mode = st.session_state.deployment_mode_radio.lower()
        # Skip if the radio value matches what we already synced from the backend —
        # this avoids a false trigger when Streamlit detects a programmatic state change.
        current = st.session_state.deployment_mode_selection.lower()
        if new_mode == current:
            return
        result = update_deployment_mode(new_mode)
        if result:
            st.session_state.deployment_mode_selection = st.session_state.deployment_mode_radio
            st.session_state["_mode_msg"] = (
                "success",
                f"Deployment mode set to **{st.session_state.deployment_mode_radio}**.",
            )
        else:
            st.session_state["_mode_msg"] = ("error", "Failed to update deployment mode.")
        st.session_state["_pending_tab"] = _TAB_INDEX

    modes = ["Production", "Simulator"]
    st.radio(
        "YAML generation target",
        modes,
        index=modes.index(st.session_state.deployment_mode_selection),
        horizontal=True,
        key="deployment_mode_radio",
        on_change=_on_mode_change,
        help="Production uses real vLLM with GPU resources. "
        "Simulator uses the vLLM simulator (no GPU required).",
    )

    mode_msg = st.session_state.pop("_mode_msg", None)
    if mode_msg:
        level, text = mode_msg
        if level == "success":
            st.success(text)
        else:
            st.error(text)

    st.divider()

    # --- Estimated Performance ---
    st.subheader("Estimated Performance")

    def _on_estimated_change():
        st.session_state.enable_estimated = st.session_state._enable_estimated_toggle

    st.toggle(
        "Enable estimated performance for models without benchmarks",
        value=st.session_state.get("enable_estimated", True),
        key="_enable_estimated_toggle",
        on_change=_on_estimated_change,
        help="When enabled, the roofline model generates synthetic performance estimates "
        "for model/GPU combinations that lack benchmark data.",
    )

    st.divider()

    # --- Quality Benchmarks ---
    st.subheader("Quality Benchmarks")
    st.caption(
        "Quality scores are computed from LM Arena and Artificial Analysis data. "
        "Enable auto-update to fetch the latest data periodically."
    )

    # Fetch current status from backend API
    try:
        status = requests.get(f"{API_BASE_URL}/api/v1/quality/auto-update", timeout=5).json()
        auto_update_enabled = status.get("enabled", False)
        arena_updated = status.get("arena_last_updated")
        aa_updated = status.get("aa_last_updated")
        arena_count = status.get("arena_model_count", 0)
        aa_count = status.get("aa_model_count", 0)
    except Exception:
        auto_update_enabled = False
        arena_updated = aa_updated = None
        arena_count = aa_count = 0

    new_value = st.toggle(
        "Auto-update quality benchmarks",
        value=auto_update_enabled,
        help="Automatically fetch the latest quality benchmark data when stale (>24h)",
    )

    if new_value != auto_update_enabled:
        try:
            requests.put(
                f"{API_BASE_URL}/api/v1/quality/auto-update",
                json={"enabled": new_value},
                timeout=5,
            )
            st.success(f"Quality auto-update {'enabled' if new_value else 'disabled'}.")
        except Exception:
            st.error("Failed to update auto-update setting.")

    # Show source status
    if arena_updated or aa_updated:
        status_parts = []
        if arena_count:
            status_parts.append(f"Arena: {arena_count:,} models")
        if aa_count:
            status_parts.append(f"AA: {aa_count:,} models")
        st.text(" | ".join(status_parts))
        if arena_updated:
            st.caption(f"Arena last updated: {arena_updated}")
        if aa_updated:
            st.caption(f"AA last updated: {aa_updated}")

    # Refresh Now button
    if st.button("Refresh Quality Data Now"):
        with st.spinner("Refreshing quality benchmarks..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/api/v1/quality/refresh",
                    timeout=120,
                )
                if resp.ok:
                    result = resp.json()
                    st.success(
                        f"Refreshed: {result.get('arena_rows', 0):,} Arena rows, "
                        f"{result.get('aa_models', 0):,} AA models"
                    )
                else:
                    st.error("Refresh failed — check backend logs.")
            except Exception as e:
                st.error(f"Refresh failed: {e}")

    st.divider()

    # --- Benchmark Database ---
    st.subheader("Benchmark Database")

    # Admin lock: when DB_ADMIN_PASSWORD is set, show a lock button.
    # Clicking it reveals a password field; correct password unlocks.
    # When no password is configured, everything is unlocked by default.
    admin_required = is_db_admin_required()
    admin_password = None
    locked = False

    if admin_required:
        unlocked = st.session_state.get("_db_unlocked", False)
        showing_input = st.session_state.get("_db_show_password", False)

        if not unlocked:
            locked = True
            lock_col, _ = st.columns([1, 11])
            with lock_col:
                if st.button(
                    "\U0001f510",
                    key="db_lock_btn",
                    type="secondary",
                    help="Click to enter admin password",
                ):
                    st.session_state["_db_show_password"] = not showing_input
                    st.rerun()

            if showing_input:
                pw = st.text_input(
                    "Admin password",
                    type="password",
                    key="db_admin_password_input",
                )
                if pw:
                    if verify_db_admin_password(pw):
                        st.session_state["_db_unlocked"] = True
                        st.session_state["_db_admin_password"] = pw
                        st.session_state["_db_show_password"] = False
                        st.rerun()
                    else:
                        st.error("Incorrect password.")
        else:
            locked = False
            admin_password = st.session_state.get("_db_admin_password")
            lock_col, _ = st.columns([1, 11])
            with lock_col:
                if st.button(
                    "\U0001f513",
                    key="db_unlock_btn",
                    type="secondary",
                    help="Click to lock admin access",
                ):
                    st.session_state["_db_unlocked"] = False
                    st.session_state.pop("_db_admin_password", None)
                    st.rerun()

    # Reserve space for stats — populated after actions so data is always fresh
    status_area = st.container()

    st.divider()

    # Track whether an action produced updated stats
    action_status = None

    # --- Upload ---
    st.markdown("**Upload Benchmarks**")
    st.caption("Upload a JSON file with a top-level `benchmarks` array. Duplicates are skipped.")

    # Counter-based key resets the file uploader after a successful load
    upload_counter = st.session_state.get("_upload_counter", 0)
    uploaded = st.file_uploader(
        "Choose benchmark JSON file",
        type=["json"],
        key=f"settings_file_upload_{upload_counter}",
        label_visibility="collapsed",
        disabled=locked,
    )

    # Clear any stored message when the user selects a new file
    if uploaded is not None:
        st.session_state.pop("_load_msg", None)

    if uploaded is not None and st.button(
        "Load DB",
        key="settings_upload_btn",
        type="primary",
        disabled=locked,
    ):
        with st.spinner("Loading..."):
            result = upload_benchmarks(uploaded.getvalue(), uploaded.name, password=admin_password)
        if result and result.get("success"):
            msg = (
                f"Processed {result.get('records_in_file', '?')} records from "
                f"{result.get('filename', 'file')} (duplicates skipped). "
                f"Database now has {result.get('total_benchmarks', '?')} unique benchmarks."
            )
            st.session_state["_load_msg"] = ("success", msg)
            # Increment counter so the file uploader resets on next rerun
            st.session_state["_upload_counter"] = upload_counter + 1
            st.session_state["_pending_tab"] = _TAB_INDEX
            st.rerun()
        else:
            msg = result.get("message", "Unknown error") if result else "No response from server"
            st.session_state["_load_msg"] = ("error", f"Load failed: {msg}")

    # Show persisted load message (survives the rerun that clears the file uploader)
    load_msg = st.session_state.get("_load_msg")
    if load_msg:
        level, text = load_msg
        if level == "success":
            st.success(text)
        else:
            st.error(text)

    st.divider()

    # --- Reset ---
    st.markdown("**Reset Database**")
    if st.button(
        "Reset Database",
        key="settings_reset_btn",
        type="secondary",
        disabled=locked,
    ):
        st.session_state["_pending_tab"] = _TAB_INDEX
        with st.spinner("Resetting..."):
            result = reset_database(password=admin_password)
        if result and result.get("success"):
            st.success("Database has been reset. All benchmark data removed.")
            action_status = result
            # Clear any stale load message
            st.session_state.pop("_load_msg", None)
        else:
            msg = result.get("message", "Unknown error") if result else "No response from server"
            st.error(f"Reset failed: {msg}")

    # --- Populate status area (after actions, so stats reflect mutations) ---
    status = action_status if action_status else fetch_db_status()
    with status_area:
        if status and status.get("success"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Benchmarks", status.get("total_benchmarks", 0))
            c2.metric("Models", status.get("num_models", 0))
            c3.metric("Hardware Types", status.get("num_hardware_types", 0))

            traffic = status.get("traffic_distribution", [])
            if traffic:
                st.caption(
                    "Traffic profiles: "
                    + ", ".join(f"({t['prompt_tokens']}, {t['output_tokens']})" for t in traffic)
                )
        else:
            st.warning("Could not connect to database.")

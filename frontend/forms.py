# forms.py

from __future__ import annotations

import streamlit as st

from components.dynamic_inputs import dynamic_list_input


def render_project_form() -> None:

    st.subheader("📋 Project Proposal")

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.proposal_title = st.text_input(
            "Project Title",
            value=st.session_state.proposal_title,
        )

    with col2:
        wk, team = st.columns(2)

        with wk:
            st.session_state.timeline_weeks = st.number_input(
                "Timeline (weeks)",
                min_value=1,
                value=st.session_state.timeline_weeks,
            )

        with team:
            st.session_state.team_size = st.number_input(
                "Team Size",
                min_value=1,
                value=st.session_state.team_size,
            )

    st.session_state.proposal_desc = st.text_area(
        "Project Description",
        value=st.session_state.proposal_desc,
        height=140,
    )

    with st.expander("📝 Requirements", expanded=True):
        st.session_state.requirements = dynamic_list_input(
            label="Requirement",
            session_key="requirements",
            placeholder="Enter requirement...",
        )

    with st.expander("⚠️ Constraints"):
        st.session_state.constraints =dynamic_list_input(
            label="Constraint",
            session_key="constraints",
            placeholder="Enter constraint...",
        )

    with st.expander("🛠️ Tech Stack"):
        st.session_state.tech_stack = dynamic_list_input(
            label="Technology",
            session_key="tech_stack",
            placeholder="e.g. FastAPI",
        )

    st.session_state.budget_range = st.text_input(
        "Budget Range",
        value=st.session_state.budget_range,
    )
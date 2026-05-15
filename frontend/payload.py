# payload.py

from __future__ import annotations

import streamlit as st


def build_payload() -> dict:
    return {
        "proposal": {
            "title": st.session_state.proposal_title,
            "description": st.session_state.proposal_desc,
            "requirements": st.session_state.requirements,
            "constraints": st.session_state.constraints,
            "timeline_weeks": int(st.session_state.timeline_weeks),
            "team_size": int(st.session_state.team_size),
            "tech_stack": st.session_state.tech_stack,
            "budget_range": st.session_state.budget_range or None,
        },
        "prompt": {
            "user_prompt": st.session_state.user_prompt_txt,
            "diagram_types": st.session_state.diagram_types_sel,
            "priority": st.session_state.priority,
            "optimize_prompt": st.session_state.optimize_prompt,
            "include_gantt": st.session_state.include_gantt,
            "include_parallel_work": True,
        },
        "hf_token": st.session_state.hf_token,   
    }
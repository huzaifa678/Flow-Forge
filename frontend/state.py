# state.py

from __future__ import annotations
import uuid

import streamlit as st


DEFAULT_STATE = {
    "proposal_title": "",
    "proposal_desc": "",
    "requirements": [],
    "constraints": [],
    "tech_stack": [],
    "budget_range": "",
    "timeline_weeks": 12,
    "team_size": 5,
    "user_prompt_txt": "",
    "hf_token": "",
    "priority": "medium",
    "optimize_prompt": True,
    "include_gantt": True,
    "diagram_types_sel": ["workflow", "ci_cd", "system_design", "flowchart", "architecture", "gantt"],
    "audience_type": "engineer",
    "recipient_email": "",
}


def init_session() -> None:
    for k, v in DEFAULT_STATE.items():
        st.session_state.setdefault(k, v)
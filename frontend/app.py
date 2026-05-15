# app.py

from __future__ import annotations

import base64
import json

import requests
import streamlit as st

from forms import render_project_form
from document_pdf import build_pdf
from payload import build_payload
from state import init_session

BACKEND_URL = "http://localhost:8000"


def main() -> None:

    st.set_page_config(
        page_title="FlowForge",
        layout="wide",
    )

    init_session()

    st.title("🏗️ FlowForge")

    tab1, tab2, tab3 = st.tabs(
        [
            "📋 Project",
            "🎨 Diagrams",
            "⚙️ Advanced",
        ]
    )

    with tab1:
        render_project_form()

    with tab2:
        st.text_area(
            "Prompt",
            key="user_prompt_txt",
            height=100,
        )
        st.multiselect(
            "Diagram Types",
            options=["workflow", "ci_cd", "system_design", "flowchart", "architecture", "gantt"],
            key="diagram_types_sel",
        )

    with tab3:
        st.selectbox(
            "Priority",
            ["low", "medium", "high", "critical"],
            key="priority",
        )

        st.toggle(
            "Optimize Prompt",
            key="optimize_prompt",
        )

        st.toggle(
            "Include Gantt",
            key="include_gantt",
        )

        st.session_state.hf_token = st.text_input(
            "HF Token",
            type="password",
            value=st.session_state.hf_token,
        )

    if st.button(
        "🚀 Generate Diagrams",
        type="primary",
        use_container_width=True,
    ):
        payload = build_payload()
        print(json.dumps(payload, indent=2))
        print(type(payload["proposal"]["requirements"]))
        print(type(payload["proposal"]["constraints"]))
        print(type(payload["proposal"]["tech_stack"]))
        st.json(payload)

        with st.spinner("Generating diagrams..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/generate-diagrams",
                    json=payload,
                    timeout=600,
                )
                response.raise_for_status()
                data = response.json()
                print(response.status_code)
                print(response.text)
                print(data)

                diagrams = data.get("diagrams", [])

                for diagram in diagrams:
                    name = diagram.get("name", "diagram")
                    image_base64 = diagram.get("image_data")
                    if image_base64:
                        if image_base64.startswith("data:image"):
                            image_base64 = image_base64.split(",", 1)[1]
                            image_bytes = base64.b64decode(image_base64)
                            st.image(image_bytes, caption=name, use_column_width=True)

                pdf_buffer = build_pdf(diagrams)

                st.download_button(
                    label="📄 Download Diagram Report (PDF)",
                    data=pdf_buffer,
                    file_name="flowforge_diagrams.pdf",
                    mime="application/pdf"
                )
            except requests.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")


if __name__ == "__main__":
    main()
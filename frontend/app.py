# app.py

from __future__ import annotations

import base64
import json

import requests
import streamlit as st

from forms import render_project_form
from document_pdf import build_pdf, build_stakeholder_pdf
from email_sender import send_pdf_email
from payload import build_payload
from state import init_session
from config import Config

BACKEND_URL = Config.BACKEND_URL

_ENGINEER_DIAGRAMS = ["workflow", "ci_cd", "system_design", "flowchart", "architecture", "gantt"]
_STAKEHOLDER_DIAGRAMS = ["flowchart", "architecture", "gantt"]


def _audience_label(audience_type: str) -> str:
    return "👷 Engineer" if audience_type == "engineer" else "📊 Stakeholder"


def main() -> None:

    st.set_page_config(
        page_title="FlowForge",
        layout="wide",
    )

    init_session()

    st.title("🏗️ FlowForge")

    st.markdown("### Who is this report for?")
    audience_choice = st.radio(
        "Audience",
        options=["engineer", "stakeholder"],
        format_func=_audience_label,
        horizontal=True,
        key="audience_type",
        help=(
            "**Engineer** — full technical diagrams (CI/CD, system design, workflow, etc.)\n\n"
            "**Stakeholder** — business-friendly diagrams with condensed spec document (project phases, timeline, solution overview)"
        ),
    )

    is_stakeholder = audience_choice == "stakeholder"

    st.divider()

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

        if is_stakeholder:
            st.info(
                "**Stakeholder mode** — diagrams are automatically filtered to "
                "business-friendly types: Project Phases, Solution Overview, Project Timeline."
            )
            # Force stakeholder diagram subset into session state
            st.session_state.diagram_types_sel = _STAKEHOLDER_DIAGRAMS
        else:
            st.multiselect(
                "Diagram Types",
                options=_ENGINEER_DIAGRAMS,
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

        with st.spinner("Generating diagrams..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/generate-diagrams",
                    json=payload,
                    timeout=600,
                )
                response.raise_for_status()
                data = response.json()

                diagrams = data.get("diagrams", [])

                for diagram in diagrams:
                    image_base64 = diagram.get("image_data")
                    if image_base64:
                        if image_base64.startswith("data:image"):
                            image_base64 = image_base64.split(",", 1)[1]
                        image_bytes = base64.b64decode(image_base64)

                        friendly = {
                            "flowchart": "Project Phases",
                            "gantt": "Project Timeline",
                            "architecture": "Solution Overview",
                        }
                        name = (
                            friendly.get(diagram.get("diagram_type", ""))
                            or diagram.get("name", "diagram")
                        )
                        st.image(image_bytes, caption=name, use_column_width=True)

                # Build PDF based on audience
                if is_stakeholder:
                    proposal_data = {
                        "title": st.session_state.proposal_title,
                        "description": st.session_state.proposal_desc,
                        "requirements": st.session_state.requirements,
                        "constraints": st.session_state.constraints,
                        "timeline_weeks": st.session_state.timeline_weeks,
                        "team_size": st.session_state.team_size,
                        "budget_range": st.session_state.budget_range or None,
                        "tech_stack": st.session_state.tech_stack,
                        "priority": st.session_state.priority,
                    }
                    pdf_buffer = build_stakeholder_pdf(diagrams, proposal_data)
                    pdf_label = "📄 Download Stakeholder Report (PDF)"
                    pdf_filename = "flowforge_stakeholder_report.pdf"
                else:
                    pdf_buffer = build_pdf(diagrams)
                    pdf_label = "📄 Download Engineering Report (PDF)"
                    pdf_filename = "flowforge_engineering_report.pdf"

                # Store PDF in session so the email section can reuse it
                st.session_state["_last_pdf"] = pdf_buffer.getvalue()
                st.session_state["_last_pdf_filename"] = pdf_filename

                st.download_button(
                    label=pdf_label,
                    data=pdf_buffer,
                    file_name=pdf_filename,
                    mime="application/pdf",
                )

            except requests.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")

    # ── Email delivery section ────────────────────────────────────────────
    if st.session_state.get("_last_pdf"):
        st.divider()
        st.subheader("📧 Send Report via Email")

        if not Config.smtp_configured():
            st.warning(
                "Email delivery is not configured. "
                "Add **SMTP_HOST**, **SMTP_USER**, and **SMTP_PASSWORD** to your `.env` file and restart the frontend."
            )
        else:
            col_email, col_btn = st.columns([3, 1])

            with col_email:
                recipient = st.text_input(
                    "Recipient Email",
                    value=st.session_state.recipient_email,
                    placeholder="name@company.com",
                    key="recipient_email",
                )

            with col_btn:
                st.write("")  # vertical align
                st.write("")
                send_clicked = st.button("Send Email", use_container_width=True)

            if send_clicked:
                if not recipient or "@" not in recipient:
                    st.error("Please enter a valid email address.")
                else:
                    with st.spinner("Sending email..."):
                        try:
                            import io
                            import smtplib
                            import socket
                            pdf_buf = io.BytesIO(st.session_state["_last_pdf"])
                            send_pdf_email(
                                recipient=recipient,
                                pdf_buffer=pdf_buf,
                                project_title=st.session_state.proposal_title or "FlowForge Project",
                                audience_type=st.session_state.audience_type,
                            )
                            st.success(f"Report sent to {recipient} ✓")
                        except socket.gaierror:
                            st.error(
                                "Could not reach the mail server. "
                                "Check that **SMTP_HOST** is correct in your `.env` (e.g. `smtp.gmail.com`)."
                            )
                        except smtplib.SMTPAuthenticationError:
                            st.error(
                                "Login rejected by the mail server. "
                                "Double-check **SMTP_USER** and **SMTP_PASSWORD**. "
                                "For Gmail, use an [App Password](https://myaccount.google.com/apppasswords) "
                                "if 2-Step Verification is on."
                            )
                        except smtplib.SMTPRecipientsRefused:
                            st.error(f"The address **{recipient}** was refused by the server. Check it and try again.")
                        except smtplib.SMTPException as exc:
                            st.error(f"Mail server error: {exc}")
                        except Exception as exc:
                            st.error(f"Unexpected error while sending email: {exc}")


if __name__ == "__main__":
    main()

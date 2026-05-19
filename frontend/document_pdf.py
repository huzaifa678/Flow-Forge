import base64
from datetime import date
from io import BytesIO
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Image as RLImage,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def build_pdf(diagrams):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    elements = []

    order = [
        "workflow",
        "ci_cd",
        "system_design",
        "flowchart",
        "architecture",
        "gantt",
    ]

    diagrams_sorted = sorted(
        diagrams,
        key=lambda x: (
            order.index(x.get("diagram_type", ""))
            if x.get("diagram_type", "") in order
            else 999
        ),
    )

    MAX_WIDTH = 500
    MAX_HEIGHT = 700

    for diagram in diagrams_sorted:
        name = diagram.get("title") or diagram.get("diagram_type", "diagram")

        image_base64 = diagram.get("image_data")

        if not image_base64:
            continue

        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_base64)

        pil_image = Image.open(BytesIO(image_bytes))
        original_width, original_height = pil_image.size

        ratio = min(
            MAX_WIDTH / original_width,
            MAX_HEIGHT / original_height,
        )

        new_width = original_width * ratio
        new_height = original_height * ratio

        img = RLImage(
            BytesIO(image_bytes),
            width=new_width,
            height=new_height,
        )

        elements.append(
            Paragraph(
                f"<b>{name}</b>",
                styles["Heading1"],
            )
        )

        elements.append(Spacer(1, 20))

        elements.append(img)

        elements.append(Spacer(1, 40))

        elements.append(PageBreak())

    doc.build(elements)

    buffer.seek(0)

    return buffer


def build_stakeholder_pdf(diagrams: list, proposal: dict) -> BytesIO:
    """Build a condensed, business-friendly spec document PDF for stakeholders."""
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontSize=28,
        spaceAfter=12,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontSize=14,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    section_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=colors.HexColor("#1A3C6E"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        spaceAfter=6,
    )
    label_style = ParagraphStyle(
        "Label",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.grey,
    )

    elements = []

    # ── Cover page ──────────────────────────────────────────────────────────
    elements.append(Spacer(1, 3 * cm))
    elements.append(Paragraph(proposal.get("title", "Project Report"), title_style))
    elements.append(Paragraph("Stakeholder Summary Report", subtitle_style))
    elements.append(Paragraph(f"Prepared: {date.today().strftime('%B %d, %Y')}", subtitle_style))
    elements.append(Spacer(1, 1 * cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1A3C6E")))
    elements.append(PageBreak())

    # ── Project overview ─────────────────────────────────────────────────────
    elements.append(Paragraph("Project Overview", section_style))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    elements.append(Spacer(1, 0.3 * cm))

    description = proposal.get("description", "")
    if description:
        elements.append(Paragraph(description, body_style))
    elements.append(Spacer(1, 0.5 * cm))

    # ── Key metrics table ────────────────────────────────────────────────────
    elements.append(Paragraph("Key Project Metrics", section_style))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    elements.append(Spacer(1, 0.3 * cm))

    metrics = [
        ["Timeline", f"{proposal.get('timeline_weeks', '—')} weeks"],
        ["Team Size", f"{proposal.get('team_size', '—')} people"],
        ["Budget Range", proposal.get("budget_range") or "To be determined"],
        ["Priority", (proposal.get("priority") or "medium").capitalize()],
    ]
    tech = proposal.get("tech_stack") or []
    if tech:
        metrics.append(["Technology", ", ".join(tech)])

    tbl = Table(metrics, colWidths=[4 * cm, 12 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F8")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1A3C6E")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 0.5 * cm))

    # ── Requirements ─────────────────────────────────────────────────────────
    requirements = proposal.get("requirements") or []
    if requirements:
        elements.append(Paragraph("Business Requirements", section_style))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        elements.append(Spacer(1, 0.3 * cm))
        for i, req in enumerate(requirements, 1):
            elements.append(Paragraph(f"{i}. {req}", body_style))
        elements.append(Spacer(1, 0.5 * cm))

    # ── Constraints ──────────────────────────────────────────────────────────
    constraints = proposal.get("constraints") or []
    if constraints:
        elements.append(Paragraph("Constraints & Assumptions", section_style))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        elements.append(Spacer(1, 0.3 * cm))
        for c in constraints:
            elements.append(Paragraph(f"• {c}", body_style))
        elements.append(Spacer(1, 0.5 * cm))

    elements.append(PageBreak())

    # ── Diagrams ─────────────────────────────────────────────────────────────
    stakeholder_order = ["flowchart", "architecture", "gantt"]
    diagrams_sorted = sorted(
        diagrams,
        key=lambda x: (
            stakeholder_order.index(x.get("diagram_type", ""))
            if x.get("diagram_type", "") in stakeholder_order
            else 999
        ),
    )

    MAX_WIDTH = 470
    MAX_HEIGHT = 650

    for diagram in diagrams_sorted:
        image_base64 = diagram.get("image_data")
        if not image_base64:
            continue

        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_base64)
        pil_image = Image.open(BytesIO(image_bytes))
        orig_w, orig_h = pil_image.size
        ratio = min(MAX_WIDTH / orig_w, MAX_HEIGHT / orig_h)

        diagram_type = diagram.get("diagram_type", "")
        friendly_names = {
            "flowchart": "Project Phases",
            "gantt": "Project Timeline",
            "architecture": "Solution Overview",
        }
        name = friendly_names.get(diagram_type, diagram.get("title", diagram_type))

        elements.append(Paragraph(name, section_style))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        elements.append(Spacer(1, 0.4 * cm))
        elements.append(RLImage(BytesIO(image_bytes), width=orig_w * ratio, height=orig_h * ratio))
        elements.append(Spacer(1, 0.5 * cm))
        elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer
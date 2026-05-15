import base64
from io import BytesIO
from PIL import Image
from reportlab.platypus import (
    SimpleDocTemplate,
    Image as RLImage,
    Paragraph,
    Spacer,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet


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
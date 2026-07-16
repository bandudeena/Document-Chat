"""Generate the demo documents in sample_docs/.

Creates a fictional company report PDF containing prose, a ruled table and an
embedded chart image — one file that exercises all three ingestion paths —
plus a plain-text policy document.

Usage:  python scripts/make_sample_docs.py
"""

import io
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parents[2] / "sample_docs"


def make_chart_png() -> bytes:
    """A simple bar chart drawn with PIL (no matplotlib dependency)."""
    width, height = 640, 360
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    bars = [
        ("Q1", 210, "#0f766e"),
        ("Q2", 260, "#0f766e"),
        ("Q3", 235, "#b45309"),
        ("Q4", 310, "#0f766e"),
    ]
    max_value = 350
    bar_width = 90
    draw.text((20, 12), "Aurora Coffee — Quarterly revenue (k EUR), 2025", fill="black")
    for i, (label, value, color) in enumerate(bars):
        x0 = 60 + i * (bar_width + 50)
        bar_height = int((value / max_value) * 250)
        y0, y1 = 320 - bar_height, 320
        draw.rectangle([x0, y0, x0 + bar_width, y1], fill=color)
        draw.text((x0 + 30, y0 - 18), str(value), fill="black")
        draw.text((x0 + 35, 328), label, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_report_pdf() -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT / "aurora-coffee-annual-report.pdf"), pagesize=A4)

    table = Table(
        [
            ["Store", "City", "Revenue 2025 (k EUR)", "Growth vs 2024"],
            ["Aurora Central", "Amsterdam", "412", "+12%"],
            ["Aurora Canal", "Amsterdam", "298", "+4%"],
            ["Aurora Hbf", "Berlin", "351", "+19%"],
            ["Aurora Old Town", "Prague", "154", "-3%"],
        ],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )

    doc.build(
        [
            Paragraph("Aurora Coffee — Annual Report 2025", styles["Title"]),
            Paragraph(
                "Aurora Coffee operates four specialty coffee stores across Amsterdam, "
                "Berlin and Prague. 2025 was our strongest year so far: total revenue "
                "reached 1,215k EUR, up 9% year over year, driven primarily by the new "
                "Berlin Hauptbahnhof location which grew 19%. The Prague Old Town store "
                "declined 3% due to street construction that blocked foot traffic between "
                "April and August.",
                styles["BodyText"],
            ),
            Spacer(1, 8),
            Paragraph("Revenue by store", styles["Heading2"]),
            table,
            Spacer(1, 12),
            Paragraph("Quarterly trend", styles["Heading2"]),
            Paragraph(
                "The chart below shows quarterly revenue. The Q3 dip reflects the summer "
                "slowdown; Q4 set a company record at 310k EUR.",
                styles["BodyText"],
            ),
            RLImage(io.BytesIO(make_chart_png()), width=14 * cm, height=7.9 * cm),
            Spacer(1, 8),
            Paragraph(
                "Outlook: in 2026 we plan to open two more stores (Rotterdam and Leipzig) "
                "and expand the wholesale bean business, targeting 1,500k EUR total revenue.",
                styles["BodyText"],
            ),
        ]
    )


def make_policy_txt() -> None:
    (OUT / "aurora-employee-handbook.txt").write_text(
        """Aurora Coffee — Employee Handbook (excerpt)

Working hours
Baristas work in two shifts: opening (06:30-14:30) and closing (13:30-21:30).
Shift swaps must be agreed with the store manager at least 48 hours in advance.

Leave policy
Full-time employees receive 25 vacation days per year plus public holidays.
Unused days can be carried over until the end of March of the following year.
Sick leave requires a doctor's note from the third consecutive day.

Free coffee policy
Employees may drink coffee freely during shifts and take home 250g of beans
per week. The staff discount for food is 50% during shifts and 25% otherwise.

Training
Every new barista completes the two-week Aurora Academy programme, covering
espresso fundamentals, milk chemistry, latte art and customer service. A
sensory calibration session is held monthly in each store.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    make_report_pdf()
    make_policy_txt()
    print(f"wrote sample docs to {OUT}")

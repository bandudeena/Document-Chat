"""Convert pdfplumber table output (list of rows of cells) to GitHub Markdown.

Markdown is chosen deliberately: the table is stored as one atomic chunk, so
the LLM sees intact rows/columns and can reproduce the table in its answer,
which the frontend renders back as a real table.
"""


def _clean_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    # newlines break Markdown table rows; pipes break column boundaries
    return " ".join(cell.split()).replace("|", "\\|")


def table_to_markdown(rows: list[list[str | None]]) -> str:
    """Render extracted table rows as a Markdown table.

    Returns "" for tables with no usable content (all-empty cells).
    Ragged rows are padded to the widest row. The first row is treated as the
    header, which matches how pdfplumber returns most real-world tables.
    """
    cleaned = [[_clean_cell(c) for c in row] for row in rows if row is not None]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    if width == 0:
        return ""
    padded = [row + [""] * (width - len(row)) for row in cleaned]

    header, *body = padded
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)

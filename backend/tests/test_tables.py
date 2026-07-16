from app.rag.tables import table_to_markdown


def test_basic_table():
    md = table_to_markdown([["Name", "Age"], ["Ada", "36"], ["Alan", "41"]])
    lines = md.splitlines()
    assert lines[0] == "| Name | Age |"
    assert lines[1] == "| --- | --- |"
    assert "| Ada | 36 |" in lines
    assert len(lines) == 4


def test_none_and_empty_cells_are_normalised():
    md = table_to_markdown([["A", None], [None, "x"]])
    assert "| A |  |" in md
    assert "|  | x |" in md


def test_ragged_rows_are_padded():
    md = table_to_markdown([["A", "B", "C"], ["1"]])
    assert md.splitlines()[2] == "| 1 |  |  |"


def test_newlines_and_pipes_in_cells():
    md = table_to_markdown([["Header"], ["line1\nline2 | with pipe"]])
    assert "line1 line2 \\| with pipe" in md
    # cell content must stay on one Markdown row
    assert len(md.splitlines()) == 3


def test_empty_table_returns_empty_string():
    assert table_to_markdown([]) == ""
    assert table_to_markdown([[None, ""], ["", None]]) == ""

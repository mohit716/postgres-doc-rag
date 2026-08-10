"""Tests for the shared normalisation layer.

These matter because HTML and PDF extraction only converge if both sides are
normalised identically.
"""

from __future__ import annotations

from pgdocrag.normalize import (
    dehyphenate,
    markdown_table,
    normalize_code,
    normalize_prose,
    split_section_number,
    title_key,
)


def test_curly_quotes_and_ligatures_fold_to_ascii():
    assert normalize_prose("\u201cquoted\u201d and \u2018single\u2019") == '"quoted" and \'single\''
    assert normalize_prose("con\ufb01guration") == "configuration"


def test_non_breaking_space_becomes_a_normal_space():
    assert normalize_prose("shared\u00a0buffers") == "shared buffers"


def test_soft_hyphen_and_zero_width_characters_are_removed():
    assert normalize_prose("data\u00adbase\u200b") == "database"


def test_pdf_line_wrapping_is_rejoined():
    assert normalize_prose("first\nsecond", from_pdf=True) == "first second"


def test_lowercase_continuation_drops_the_line_break_hyphen():
    assert dehyphenate("config-\nuration") == "configuration"


def test_hyphen_survives_only_where_the_case_changes():
    assert dehyphenate("pre-\nPostgreSQL") == "pre-PostgreSQL"


def test_all_caps_keyword_split_across_lines_is_rejoined():
    """The PDF wraps SQL keywords mid-word: "CRE-\\nATE INDEX"."""
    assert dehyphenate("CRE-\nATE INDEX") == "CREATE INDEX"
    assert dehyphenate("CON-\nCURRENTLY") == "CONCURRENTLY"


def test_code_keeps_line_structure_and_is_dedented():
    code = "    SELECT 1;\n      FROM t;\n"
    assert normalize_code(code) == "SELECT 1;\n  FROM t;"


def test_section_numbering_is_split_from_the_title():
    assert split_section_number("19.3.1. Connection Settings") == ("19.3.1", "Connection Settings")
    assert split_section_number("Chapter 20. Client Authentication") == ("20", "Client Authentication")
    assert split_section_number("F.1. adminpack") == ("F.1", "adminpack")


def test_command_names_are_not_mistaken_for_numbering():
    assert split_section_number("ALTER TABLE") == (None, "ALTER TABLE")


def test_title_key_ignores_numbering_and_punctuation():
    assert title_key("19.3. Connections and Authentication") == title_key(
        "Connections and Authentication"
    )


def test_markdown_table_pads_ragged_rows_and_escapes_pipes():
    rendered = markdown_table([["a", "b"], ["1"]])
    assert rendered.splitlines() == ["| a | b |", "| --- | --- |", "| 1 |  |"]
    assert "\\|" in markdown_table([["a|b"], ["c"]])

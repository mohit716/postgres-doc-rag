"""Tests for the section-aware chunker.

The token ceiling is the load-bearing invariant: a chunk longer than the
embedding model's window is truncated silently, so its tail would never be
searchable.
"""

from __future__ import annotations

import pytest

from pgdocrag import config
from pgdocrag.chunk.sectioner import chunk_document
from pgdocrag.schema import CODE, PARAMETER, PROSE, Block, Document, Section

SENTENCE = "The server writes a checkpoint record to the write ahead log. "


def make_document(sections: list[Section]) -> Document:
    return Document(
        doc_id="test.html",
        source_format="html",
        pg_version=config.PG_VERSION,
        title="Server Configuration",
        sections=sections,
        source_url="https://example.test/test.html",
    )


def make_section(blocks: list[Block], **overrides) -> Section:
    defaults = dict(
        title="Connection Settings",
        level=2,
        path=["Server Configuration", "Connection Settings"],
        blocks=blocks,
    )
    defaults.update(overrides)
    return Section(**defaults)


def test_chunk_text_starts_with_the_breadcrumb():
    document = make_document([make_section([Block(kind=PROSE, text=SENTENCE * 3)])])
    chunk = chunk_document(document)[0]
    assert chunk.text.startswith("Server Configuration > Connection Settings\n\n")
    assert chunk.section_path == ["Server Configuration", "Connection Settings"]


def test_no_chunk_exceeds_the_model_window():
    document = make_document([make_section([Block(kind=PROSE, text=SENTENCE * 400)])])
    chunks = chunk_document(document)
    assert len(chunks) > 1
    assert all(chunk.token_count <= config.MODEL_MAX_TOKENS for chunk in chunks)


def test_oversized_code_block_is_split_into_balanced_fences():
    code = "```sql\n" + "SELECT column_name FROM some_table WHERE id = 1;\n" * 200 + "```"
    document = make_document([make_section([Block(kind=CODE, text=code)])])
    chunks = chunk_document(document)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= config.MODEL_MAX_TOKENS
        assert chunk.text.count("```") % 2 == 0


def test_anchor_is_appended_to_the_source_url():
    document = make_document(
        [make_section([Block(kind=PROSE, text=SENTENCE)], anchor="GUC-PORT")]
    )
    chunk = chunk_document(document)[0]
    assert chunk.source_url.endswith("#GUC-PORT")
    assert chunk.section_anchor == "GUC-PORT"


def test_parameter_sections_are_never_merged_away():
    """Merging a parameter would cost its anchor, and with it the exact citation."""
    sections = [
        make_section(
            [Block(kind=PROSE, text="Sets the port.")],
            title="port (integer)",
            path=["Server Configuration", "port (integer)"],
            anchor="GUC-PORT",
            kind=PARAMETER,
        ),
        make_section(
            [Block(kind=PROSE, text="Sets the host.")],
            title="listen_addresses (string)",
            path=["Server Configuration", "listen_addresses (string)"],
            anchor="GUC-LISTEN-ADDRESSES",
            kind=PARAMETER,
        ),
    ]
    chunks = chunk_document(make_document(sections))
    assert len(chunks) == 2
    assert {chunk.section_anchor for chunk in chunks} == {"GUC-PORT", "GUC-LISTEN-ADDRESSES"}


def test_tiny_trailing_section_is_merged_into_its_predecessor():
    sections = [
        make_section([Block(kind=PROSE, text=SENTENCE * 4)], title="Overview"),
        make_section([Block(kind=PROSE, text="See also.")], title="Notes"),
    ]
    chunks = chunk_document(make_document(sections))
    assert len(chunks) == 1
    assert "See also." in chunks[0].text


def test_sections_without_content_produce_no_chunks():
    document = make_document([make_section([], title="Parameters")])
    assert chunk_document(document) == []


def test_chunk_ids_are_stable_across_runs():
    document = make_document([make_section([Block(kind=PROSE, text=SENTENCE * 2)])])
    assert [c.chunk_id for c in chunk_document(document)] == [
        c.chunk_id for c in chunk_document(document)
    ]


@pytest.mark.parametrize(
    "blocks, expected",
    [
        ([Block(kind=PROSE, text=SENTENCE)], PROSE),
        ([Block(kind=CODE, text="```\nSELECT 1;\n```")], CODE),
        ([Block(kind=PROSE, text=SENTENCE), Block(kind=CODE, text="```\nSELECT 1;\n```")], "mixed"),
    ],
)
def test_chunk_type_reflects_block_composition(blocks, expected):
    document = make_document([make_section(blocks)])
    assert chunk_document(document)[0].chunk_type == expected

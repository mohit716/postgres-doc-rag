"""Emit the structural skeleton of the PDF manual.

The PDF counterpart to probe_structure.py: reports the bookmark outline, font
inventory, and the vertical position of repeated page furniture, which is what
the PDF extractor needs in order to reconstruct section hierarchy and strip
running headers and footers.

Usage:
    python scripts/probe_pdf.py data/raw/pdf/postgresql-18-A4.pdf
    python scripts/probe_pdf.py data/raw/pdf/postgresql-18-A4.pdf --page 640
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pymupdf as fitz


def report_outline(doc: fitz.Document) -> None:
    toc = doc.get_toc(simple=True)
    print(f"\n== outline: {len(toc)} entries ==")
    levels = Counter(level for level, _title, _page in toc)
    for level in sorted(levels):
        print(f"  level {level}: {levels[level]} entries")
    print("  first 8 entries:")
    for level, title, page in toc[:8]:
        print(f"    L{level} p{page:<5} {title[:70]}")
    print("  a mid-document window:")
    for level, title, page in toc[len(toc) // 2 : len(toc) // 2 + 6]:
        print(f"    L{level} p{page:<5} {title[:70]}")


def report_fonts(doc: fitz.Document, sample_pages: list[int]) -> None:
    fonts: Counter[str] = Counter()
    sizes: Counter[float] = Counter()
    for page_number in sample_pages:
        page = doc[page_number]
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    fonts[span["font"]] += len(span.get("text", ""))
                    sizes[round(span["size"], 1)] += 1
    print(f"\n== fonts by character count (sample of {len(sample_pages)} pages) ==")
    for name, count in fonts.most_common(12):
        print(f"  {count:>7}  {name}")
    print("  sizes:", ", ".join(f"{size}({count})" for size, count in sizes.most_common(8)))


def page_lines(page: fitz.Page) -> list[tuple[float, float, str, str]]:
    """Return (y0, y1, text, dominant font) per line, in reading order."""
    lines: list[tuple[float, float, str, str]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            font = Counter(
                {span["font"]: len(span.get("text", "")) for span in spans}
            ).most_common(1)[0][0]
            bbox = line["bbox"]
            lines.append((bbox[1], bbox[3], text, font))
    lines.sort(key=lambda item: item[0])
    return lines


def report_short_lines(doc: fitz.Document, page_number: int) -> None:
    """Show the short lines on a page: headings live among these."""
    page = doc[page_number]
    height = page.rect.height
    print(f"\n== page {page_number + 1}: short lines (heading candidates) ==")
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(span["text"] for span in spans).strip()
            if len(text) > 46:
                continue
            fonts = "|".join(sorted({span["font"] for span in spans}))
            size = max(span["size"] for span in spans)
            y = line["bbox"][1] / height
            x = line["bbox"][0]
            print(f"  y={y:.3f} x={x:>6.1f} size={size:>4.1f} {fonts[:30]:<30} {text[:44]!r}")


def report_furniture(doc: fitz.Document, sample_pages: list[int]) -> None:
    print(f"\n== page furniture (top/bottom lines across {len(sample_pages)} pages) ==")
    heights: Counter[float] = Counter()
    top_positions: list[float] = []
    bottom_positions: list[float] = []
    top_texts: Counter[str] = Counter()
    bottom_texts: Counter[str] = Counter()

    for page_number in sample_pages:
        page = doc[page_number]
        height = page.rect.height
        heights[round(height, 1)] += 1
        lines = page_lines(page)
        if not lines:
            continue
        top_positions.append(lines[0][0] / height)
        bottom_positions.append(lines[-1][1] / height)
        top_texts["".join(c for c in lines[0][2] if not c.isdigit()).strip()[:50]] += 1
        bottom_texts["".join(c for c in lines[-1][2] if not c.isdigit()).strip()[:50]] += 1

    print(f"  page height: {heights.most_common(3)}")
    if top_positions:
        print(
            f"  first line y/height: min {min(top_positions):.3f} "
            f"max {max(top_positions):.3f}"
        )
        print(
            f"  last line  y/height: min {min(bottom_positions):.3f} "
            f"max {max(bottom_positions):.3f}"
        )
    print("  repeated top-line shapes:")
    for text, count in top_texts.most_common(5):
        print(f"    {count:>3}x {text!r}")
    print("  repeated bottom-line shapes:")
    for text, count in bottom_texts.most_common(5):
        print(f"    {count:>3}x {text!r}")


def report_page(doc: fitz.Document, page_number: int) -> None:
    page = doc[page_number]
    height = page.rect.height
    lines = page_lines(page)
    print(f"\n== page {page_number + 1}: {len(lines)} lines, height {height:.0f} ==")
    for y0, y1, text, font in lines[:6]:
        print(f"  y={y0 / height:.3f} {font[:28]:<28} {text[:60]!r}")
    print("  ...")
    for y0, y1, text, font in lines[-4:]:
        print(f"  y={y0 / height:.3f} {font[:28]:<28} {text[:60]!r}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--page", type=int, default=None, help="1-based page to dump.")
    args = parser.parse_args()

    doc = fitz.open(args.path)
    print(f"== {args.path.name}: {doc.page_count} pages ==")

    step = max(doc.page_count // 40, 1)
    sample = list(range(200, min(doc.page_count, 200 + step * 40), step))

    if args.page:
        report_page(doc, args.page - 1)
        report_short_lines(doc, args.page - 1)
        return 0

    report_outline(doc)
    report_fonts(doc, sample[:20])
    report_furniture(doc, sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


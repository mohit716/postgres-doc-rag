"""Emit the structural skeleton of a documentation page.

This exists to support structure-only parser development: it reports tags,
classes, anchors and the container tree of a page without dumping its prose, so
the extraction logic can be written (or AI-assisted) from structure alone. On
confidential corpora this output is what you share instead of the document.

Usage:
    python scripts/probe_structure.py https://www.postgresql.org/docs/18/sql-createtable.html
    python scripts/probe_structure.py data/raw/html/sql-createtable.html
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

SECTION_CLASS = re.compile(
    r"^(sect[1-5]|refsect[1-3]|refentry|refnamediv|refsynopsisdiv|chapter|"
    r"preface|appendix|part|glossdiv|bibliodiv)$"
)
TITLE_CAP = 60


def load(target: str) -> str:
    if target.startswith("http"):
        response = requests.get(
            target,
            timeout=30,
            headers={"User-Agent": "PostgresDocRAG/0.1 structure probe"},
        )
        response.raise_for_status()
        return response.text
    return Path(target).read_text(encoding="utf-8", errors="replace")


def classes_of(tag: Tag) -> list[str]:
    value = tag.get("class") or []
    return [value] if isinstance(value, str) else list(value)


def report_class_frequency(root: Tag) -> None:
    counter: Counter[str] = Counter()
    for tag in root.find_all(True):
        for cls in classes_of(tag):
            counter[f"{tag.name}.{cls}"] += 1
        if not classes_of(tag):
            counter[tag.name] += 1
    print("\n== tag/class frequency (top 30) ==")
    for name, count in counter.most_common(30):
        print(f"  {count:>4}  {name}")


def report_headings(root: Tag) -> None:
    print("\n== heading outline ==")
    for tag in root.find_all(re.compile(r"^h[1-6]$")):
        label = ".".join(classes_of(tag)) or "-"
        parent_classes = ".".join(classes_of(tag.parent)) if tag.parent else "-"
        text = tag.get_text(" ", strip=True)[:TITLE_CAP]
        anchor = tag.get("id") or ""
        print(f"  <{tag.name} class={label}> parent=div.{parent_classes} id={anchor} :: {text}")


def report_container_tree(root: Tag) -> None:
    print("\n== section container tree (class, id, direct child classes) ==")
    _walk_containers(root, 0)


def _walk_containers(root: Tag, depth: int) -> None:
    for child in root.find_all(True, recursive=False):
        child_classes = classes_of(child)
        if child.name == "div" and any(SECTION_CLASS.match(c) for c in child_classes):
            anchor = child.get("id") or "-"
            direct = Counter()
            for grandchild in child.find_all(True, recursive=False):
                key = grandchild.name
                gc_classes = classes_of(grandchild)
                if gc_classes:
                    key += "." + gc_classes[0]
                direct[key] += 1
            summary = ", ".join(f"{k}x{v}" for k, v in direct.most_common(8))
            print(f"  {'  ' * depth}div.{child_classes[0]} #{anchor} -> {summary}")
            _walk_containers(child, depth + 1)
        else:
            _walk_containers(child, depth)


def report_definition_lists(root: Tag) -> None:
    print("\n== definition lists (candidate atomic units) ==")
    for dl in root.select("dl.variablelist"):
        terms = dl.find_all("dt", recursive=True)
        anchors = [dt.get("id") for dt in terms if dt.get("id")]
        print(f"  dl.variablelist: {len(terms)} dt, {len(anchors)} with id")
        for anchor in anchors[:4]:
            print(f"    dt #{anchor}")
        if len(anchors) > 4:
            print(f"    ... {len(anchors) - 4} more")


def report_blocks(root: Tag) -> None:
    print("\n== block-level content elements ==")
    counter: Counter[str] = Counter()
    for selector in ("pre", "table", "div.note", "div.warning", "div.tip",
                     "div.caution", "div.example", "div.figure", "blockquote"):
        for tag in root.select(selector):
            label = ".".join(classes_of(tag)) or "-"
            counter[f"{tag.name}.{label}"] += 1
    for name, count in counter.most_common():
        print(f"  {count:>4}  {name}")


def main() -> int:
    # Docs contain em dashes and typographic quotes; the Windows console defaults
    # to a codepage that cannot encode them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    target = sys.argv[1]
    soup = BeautifulSoup(load(target), "lxml")
    content = soup.select_one("#docContent") or soup.body
    if content is None:
        print("No content root found")
        return 1

    print(f"== source: {target} ==")
    print(f"== content root: #docContent present: {soup.select_one('#docContent') is not None} ==")
    report_class_frequency(content)
    report_headings(content)
    report_container_tree(content)
    report_definition_lists(content)
    report_blocks(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

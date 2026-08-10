# Source structure notes

Everything in this file was produced by the two probe scripts, not by reading the
documentation itself:

```bash
python scripts/probe_structure.py https://www.postgresql.org/docs/18/sql-createtable.html
python scripts/probe_pdf.py data/raw/pdf/postgresql-18-A4.pdf --page 1903
```

Both parsers were written against these skeletons. That is the point of the
exercise: on a confidential corpus you cannot paste documents into an external
model, but you *can* share a tag inventory, a class frequency table, a font list
and a heading outline — and that is enough to generate the parser. Each finding
below maps to a specific decision in the extractors, and the probes are checked
into the repo so the notes can be regenerated when the docs change.

## HTML (`extract/html_extract.py`)

### Content root and furniture

Body content lives in `div#docContent`. Everything outside it is site chrome
(`div#pgNavbar`, the search `div.input-group`, the version selector). Inside it,
these must be removed before extraction:

| Selector | Why |
| --- | --- |
| `div.navheader`, `div.navfooter` | Prev/Up/Next navigation |
| `div.toc` | Per-page table of contents; on chapter landing pages it is the *only* content, which is why those pages correctly yield zero chunks |
| `a.id_link` | The `#` permalink glyph, which otherwise lands inside every heading |
| `a.indexterm` | Invisible index anchors |

### Two container families

This is the finding that most shaped the parser. The manual mixes two DocBook
renderings, and handling only the first silently drops the entire command
reference.

Chapter pages nest `div.sect1 > div.sect2`, with headings wrapped in
`div.titlepage` as `h2.title` / `h3.title`:

```
div.sect1 #RUNTIME-CONFIG-CONNECTION -> div.titlepage x1, div.toc x1, div.sect2 x4
  div.sect2 #RUNTIME-CONFIG-CONNECTION-SETTINGS -> div.titlepage x1, div.variablelist x1
```

Reference pages nest `div.refentry > div.refsect1 > div.refsect2`, with headings
as **bare `h2`/`h3` carrying no class at all**:

```
div.refentry #SQL-CREATETABLE -> div.titlepage x1, div.refnamediv x1,
                                 div.refsynopsisdiv x1, div.refsect1 x6
  div.refnamediv     -> h2 x1, p x1          (title + one-line summary)
  div.refsynopsisdiv -> h2 x1, pre.synopsis x1
  div.refsect1 #SQL-CREATETABLE-DESCRIPTION -> h2 x1, p x6
  div.refsect1 #id-1.9.3.85.6               -> h2 x1, div.variablelist x1, div.refsect2 x1
```

Consequences in code: heading resolution tries `div.titlepage` first, then the
`refentry`'s `refnamediv`, then a direct-child heading. `refnamediv` is
deliberately *not* treated as a section, because its heading duplicates the
parent's title; only its summary paragraph is kept.

Note that some `refsect1` anchors are content-addressed (`#id-1.9.3.85.6`) rather
than semantic. They work as URLs for this release but should not be assumed
stable across versions.

### Parameters are the atomic unit

Both families converge on `dl.variablelist`, and every `dt` carries an id:

```
dl.variablelist: 10 dt, 10 with id     dt #GUC-LISTEN-ADDRESSES, dt #GUC-PORT ...
dl.variablelist: 52 dt, 52 with id     dt #SQL-CREATETABLE-TEMPORARY ...
```

Each `dt`/`dd` pair is promoted to its own section. One configuration parameter
or one command clause becomes one retrievable chunk with a permanent citation
URL, which is both the best granularity for "how do I configure X" questions and
the source of the evaluation labels. Nested variable lists (a `dl` inside a `dd`,
such as the `LIKE` options) stay inline in their parent term to avoid duplicating
content.

### Inline markup worth keeping

`code.literal`, `code.varname`, `code.type`, `code.filename`, `code.function` and
`code.command` are frequent (371 `code.literal` on the `CREATE TABLE` page alone)
and carry real signal, so they survive as backticks in chunk text. Hyperlink
targets do not: `a.xref` contributes its text only. Titles use an unmarked
variant, since backticks around every parameter name are noise in a breadcrumb.

`pre.programlisting` is a code sample and `pre.synopsis` is a command signature;
both are emitted as fenced blocks. Admonitions (`div.note`, `div.warning`,
`div.tip`, `div.caution`) carry an `h3.title` that must not be mistaken for a
section heading.

### Crawl strategy

Every page carries navigation links annotated with `accesskey`: `p` previous, `u`
up, `h` home, `n` next. Following `n` from any seed walks the manual in reading
order, which is more reliable than parsing the sitemap and supplies a page
ordinal for free. Following `u` gives the parent page, which is what lets a
breadcrumb span pages — a page knows it is "Connection Settings" but only its
ancestors know that sits under "Server Configuration".

`robots.txt` disallows only `/docs/devel/`, so the versioned tree is crawlable;
the crawler still rate-limits to one request per second with an identifying
User-Agent.

## PDF (`extract/pdf_extract.py`)

### The outline is the hierarchy

The A4 manual is 3,130 pages with a 4,023-entry bookmark outline spanning six
levels (13 parts, 93 chapters, 956 sections, 2,219 subsections). Reading section
structure from the outline avoids inferring hierarchy from font sizes.

Its coordinates are not usable directly — the stored destination uses a different
origin than the text layer — so each entry's heading is located by matching its
title text against the lines on its page. Two y-coordinates come out of that: a
section's own content starts *below* its heading, and the previous section ends
*above* it. Using one value for both leaks every heading into the section before
it.

The outline is also incomplete: `Synopsis` on every command reference page is a
real heading with no outline entry. Those are recovered by font (see below).

### Fonts carry the semantics

Character counts across a 20-page sample:

```
36830  Times-Roman        body text, 10pt at x=120
 8441  Courier            code and parameter names, 10pt
  895  Helvetica-Bold     headings, 14.4-24.9pt at x=72
  610  Courier-Oblique    replaceable tokens inside code
  272  Times-Italic       emphasis
  137  Times-Bold         inline emphasis, NOT headings
```

So headings are Helvetica-Bold above 11.5pt — deliberately distinguished from
Times-Bold, which is inline emphasis — and code is any Courier variant.

A line is treated as code only when *every* span is monospace **and** it is part
of a run of at least two such lines. A lone monospace line is usually a
parameter name or a prose line dominated by an inline code span, not a code
block.

### Configuration parameters are mixed-font

The outline stops above parameter level, so without extra work the PDF would
produce whole-section chunks while HTML produces one chunk per parameter, making
the two corpora incomparable. Parameters are recovered by their distinctive
shape:

```
listen_addresses (string)
recovery_target_timeline (string)
```

The trap: these lines are **not** fully monospace. The name is Courier but the
surrounding parentheses are Times-Roman, so an "all spans are monospace" test
silently matches nothing. The rule used instead is: the first span is monospace
and the whole line matches `name (type)`.

The HTML anchor form is then reconstructed (`work_mem` → `GUC-WORK-MEM`) to give
the two formats a shared join key, so one gold set can score both collections.

### Page furniture

Pages are 841.9pt tall. The running chapter header sits at y/height 0.040 and the
page number at 0.940, while body text never starts above 0.083. Lines outside
0.065-0.925 are dropped.

Printed page numbers run about 39 behind PDF page indices (PDF page 1903 prints
"1864") because of front matter. Citations use the PDF index, since that is what
a viewer's page box expects.

### Paragraph reconstruction

PDF text arrives as positioned lines with no paragraph structure. Lines are
grouped into paragraphs when the vertical gap exceeds 1.55x the median line
spacing for that section, then rejoined and de-hyphenated.

## Known extraction gaps

Recorded rather than hidden, since the comparison harness measures their effect:

- **Line-break hyphens are ambiguous.** `config-\nuration` should rejoin without
  the hyphen; `non-\nblocking` should keep it, and nothing in the extracted text
  distinguishes them. The hyphen is kept only where the case changes across the
  break (`pre-\nPostgreSQL`), so `non-blocking` loses its hyphen.

  An earlier version kept the hyphen before *any* uppercase continuation, which
  corrupted every SQL keyword the PDF wrapped mid-word — retrieved passages read
  `a regular CRE-ATE INDEX command ... CON-CURRENTLY cannot`. Requiring an actual
  case *change* fixes those while still protecting real compounds.
- **Lost inter-span spaces.** The PDF text layer occasionally omits the space at a
  font boundary, yielding `and:: allows` where the HTML reads `and :: allows`.
  Inserting a space at every font change would corrupt `(string)` into
  `( string )` and break parameter detection, so this is left alone.
- **No inline code marking from PDF.** Font information is discarded when prose is
  flattened, so the PDF side cannot mark parameter names as code. The comparison
  strips backticks from both sides before measuring.
- **Table structure is not recovered from PDF.** Tables become positioned text;
  only the HTML path emits markdown tables. `colspan`/`rowspan` are flattened
  even in HTML, since markdown cannot express them.

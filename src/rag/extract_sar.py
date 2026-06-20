"""extract_sar.py — SAR collection, step 1: PDF → clean, in-lane sections.

The collection-specific FRONT of the corpus pipeline (extract + select + annotate).
Output = the kept sections, each with its metadata + contextual header; the SHARED
chunk → embed → load backend consumes this later. Design: docs/design/sar-collection-design.md.

This is a fill-in-the-blanks scaffold. The plumbing (pdfplumber load, cleaning, the
record + run loop) is done; you write the four TODO(you) blocks — the parts that
encode the SAR domain knowledge and teach you the heading-splitter idea.

Prereq: `uv add pdfplumber`

CHECKPOINT (one doc, no DB, print only): run on the humpback SAR and eyeball that you
KEEP Vessel Strikes / Status / etc. and NOT the methodology/PBR sections:

    uv run python -m rag.extract_sar
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

CORPUS = Path.home() / "Source/DATA/oceans/corpus/sar"


# ══════════════════════════════════════════════════════════════════════════════
# TODO(you) #1 — the SAR template + the keep/drop decision
#   List the section headings of a NOAA SAR as they appear in the text (from
#   sar-collection-design.md). Note: MAJOR sections are ALL-CAPS ("STATUS OF STOCK"),
#   subsections are Title-Case ("Vessel Strikes") — include both kinds you care about.
#   Then KEEP_SECTIONS = the subset we keep (the reasoning the card can't hold).
# ══════════════════════════════════════════════════════════════════════════════
KNOWN_HEADINGS: list[str] = [
    # "STOCK DEFINITION AND GEOGRAPHIC RANGE",
    # "POPULATION SIZE",
    # "HUMAN-CAUSED MORTALITY AND SERIOUS INJURY",
    # "Vessel Strikes",
    # "STATUS OF STOCK",
    # ... fill in from the design doc ...
]
KEEP_SECTIONS: set[str] = {
    # the subset to keep, e.g. "Vessel Strikes", "STATUS OF STOCK", ...
}
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Section:
    """One KEPT section of one SAR — the unit the shared chunker will consume."""
    source: str    # friendly doc name,  e.g. "Humpback CA/OR/WA SAR"
    species: str   # common name,         e.g. "Humpback Whale"  (= individuals.common_name)
    stock: str     # stock label,         e.g. "California/Oregon/Washington"
    year: int
    section: str   # the heading,         e.g. "Vessel Strikes"
    text: str      # the cleaned body
    header: str    # the contextual header (prepended before embedding)


# ── plumbing: PDF → flat text → cleaned (DONE — read, don't edit unless tuning) ──
def load_text(pdf_path: Path) -> str:
    """pdfplumber: PDF → one flat text string, page by page (no structure survives)."""
    parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def clean(text: str) -> str:
    """Strip page furniture so the heading-match and chunks aren't polluted.

    Handles the common SAR noise; tune once you see real output at the checkpoint.
    """
    # de-hyphenate words split across a line break:  "popula-\ntion" -> "population"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    out: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue            # blank line
        if s.isdigit():
            continue            # bare page number
        out.append(s)
    return "\n".join(out)
# ────────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# TODO(you) #2 — the heading splitter (THE educational core; "our" heading split)
#   The headings are NOT markup (the PDF lost them), so we supply them. Walk the
#   lines: when a line IS one of KNOWN_HEADINGS, start a new section; otherwise
#   append the line to the current section's body. Return {heading: body} for the
#   sections whose heading is in KEEP_SECTIONS.
#   (You'll discover matching is fiddly — exact match? case-normalize? — that's the
#    learning. Eyeball the checkpoint and tighten it.)
# ══════════════════════════════════════════════════════════════════════════════
def split_into_sections(text: str) -> dict[str, str]:
    raise NotImplementedError("TODO(you) #2: walk the lines, cut on KNOWN_HEADINGS, keep KEEP_SECTIONS")
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# TODO(you) #3 — filename → metadata
#   "humpback-caorwa-2021.pdf" -> {source, species, stock, year}
#   Hint: split the stem on '-'. Map the species slug ('humpback' -> 'Humpback Whale')
#   and the region slug ('caorwa' -> 'California/Oregon/Washington') via small dicts.
# ══════════════════════════════════════════════════════════════════════════════
def parse_filename(pdf_path: Path) -> dict:
    raise NotImplementedError("TODO(you) #3: derive source/species/stock/year from the filename")
# ──────────────────────────────────────────────────────────────────────────────


def extract(pdf_path: Path) -> list[Section]:
    """One SAR → list[Section]  (extract + select + annotate; no chunk/embed yet)."""
    meta = parse_filename(pdf_path)               # TODO(you) #3
    raw = clean(load_text(pdf_path))              # scaffolded
    kept = split_into_sections(raw)               # TODO(you) #2

    sections: list[Section] = []
    for heading, body in kept.items():
        section = heading.title()                 # normalize "VESSEL STRIKES" -> "Vessel Strikes"
        # ── TODO(you) #4: build the contextual header from meta["source"] + section ──
        #    e.g.  f'[{meta["source"]} — {section}]'   ->  "[Humpback CA/OR/WA SAR — Vessel Strikes]"
        header = None  # <- replace
        # ────────────────────────────────────────────────────────────────────────────
        sections.append(Section(
            source=meta["source"], species=meta["species"], stock=meta["stock"],
            year=meta["year"], section=section, text=body, header=header,
        ))
    return sections


if __name__ == "__main__":
    # CHECKPOINT — one doc, print only. Eyeball: kept sections + their headers,
    # and that the methodology/PBR sections are absent.
    pdf = CORPUS / "humpback-caorwa-2021.pdf"
    secs = extract(pdf)
    print(f"\n{pdf.name}: kept {len(secs)} sections\n")
    for s in secs:
        print(f"=== {s.header} ===")
        print(f"    ({len(s.text)} chars)  {s.text[:160].strip()} ...\n")

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
from enum import StrEnum
from pathlib import Path
from typing import Optional

import pdfplumber

CORPUS = Path.home() / "Source/DATA/oceans/corpus/sar"


# ══════════════════════════════════════════════════════════════════════════════
# TODO(you) #1 — the SAR template + the keep/drop decision
#   List the section headings of a NOAA SAR as they appear in the text (from
#   sar-collection-design.md). Note: MAJOR sections are ALL-CAPS ("STATUS OF STOCK"),
#   subsections are Title-Case ("Vessel Strikes") — include both kinds you care about.
#   Then KEEP_SECTIONS = the subset we keep (the reasoning the card can't hold).
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Group:
    source: str    # friendly doc name,  e.g. "Humpback CA/OR/WA SAR"
    species: str   # common name,         e.g. "Humpback Whale"  (= individuals.common_name)
    stock: str     # stock label,         e.g. "California/Oregon/Washington"
    year: int
    file: Optional[str] = None

SAR="NOAA Fisheries' Marine Mammal Stock Assessment Reports"

class CommonName(StrEnum):
    BLUE: "Blue Whale"
    SPERM: "Sperm Whale"
    GRAY: "Gray Whale"
    HUMPBACK: "Humpback Whale"
    FIN: "Fin Whale"
    ORCA: "Killer Whale"


FILES : dict = {
    "blue-whale-enp-2023.pdf": Group(source=SAR, species=CommonName.BLUE, stock="Eastern North Pacific", year=2024),
    "fin-whale-caorwa-2023.pdf": Group(source=SAR, species=CommonName.FIN, stock="California Oregon Washington ", year=2024),
    "gray-whale-eastern-np-2020.pdf": Group(source=SAR, species=CommonName.GRAY, stock="Eastern North Pacific", year=2021),
    "humpback-caorwa-2021.pdf": Group(source=SAR, species=CommonName.HUMPBACK, stock="California Oregon Washington", year=2022),
    "humpback-casmex-dps-2022.pdf": Group(source=SAR, species=CommonName.HUMPBACK, stock="Central America / Southern Mexico - California Oregon Washington", year=2023),
    "killer-whale-enp-2024.pdf": Group(source=SAR, species=CommonName.ORCA, stock="Eastern Pacific Southern Resident", year=2024),
    "killer-whale-southern-resident-2021.pdf": Group(source=SAR, species=CommonName.ORCA, stock="Eastern North Pacific Southern Resident", year=2022),
    "pacific-mmsar-2024-combined.pdf": Group(source=SAR, species="combined", stock="", year=0),
    "sperm-whale-caorwa-2023.pdf": Group(source=SAR, species=CommonName.SPERM, stock="California Oregon Washington", year=2024),
}


KNOWN_HEADINGS: dict = {
    "STOCK DEFINITION AND GEOGRAPHIC RANGE": "STOCK",
    "POPULATION SIZE": "STOCK",
    # "Minimum Population Estimate": "",
    # "Current Population Trend": "",
    "CURRENT AND MAXIMUM NET PRODUCTIVITY RATES": "PRODUCTIVITY",
    "POTENTIAL BIOLOGICAL REMOVAL": "STOCK",
    "HUMAN-CAUSED MORTALITY AND SERIOUS INJURY": "MORTALITY",   # "Vessel Strikes"
    "Fishery Information": None,
    "Vessel Strikes": "MORTALITY",
    "Other human-caused mortality and serious injury": "MORTALITY",
    "Habitat Concerns": "MORTALITY",
    "STATUS OF STOCK": "STOCK",
}


KEEP_SECTIONS: set[str] = {
    "STOCK DEFINITION AND GEOGRAPHIC RANGE"
}
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Section:
    """One KEPT section of one SAR — the unit the shared chunker will consume."""
    file: str    # friendly doc name,  e.g. "Humpback CA/OR/WA SAR"
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
        for i, page in enumerate(pdf.pages):
            print(f"Reading Page {i}")
            parts.append(page.extract_text() or "")
    return "\n".join(parts)

# def temp(pdf_path):
#     documents = []
#
#     with pdfplumber.open(pdf_path) as pdf:
#         for page_num, page in enumerate(pdf.pages):
#             # 1. Extract any tables on the page
#             tables = page.extract_tables()
#
#             # 2. Extract standard text
#             text = page.extract_text() or ""
#
#             # 3. Simple layout preservation: Append tables as Markdown strings
#             table_strings = ""
#             if tables:
#                 for table in tables:
#                     # Filter out empty rows
#                     cleaned_table = [row for row in table if any(row)]
#                     for row in cleaned_table:
#                         # Convert list row to markdown row syntax
#                         table_strings += "| " + " | ".join([str(cell or "") for cell in row]) + " |\n"
#                     table_strings += "\n"
#
#             # Combine page contents with basic metadata
#             page_content = f"{text}\n\n### Extracted Tables:\n{table_strings}"
#
#             documents.append({
#                 "page_content": page_content,
#                 "metadata": {"source": pdf_path, "page": page_num + 1}
#             })
#
#     return documents



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
def split_into_sections(path: Path, text: str) -> list[Section]:
    normalized = re.sub(r'\s+', ' ', text)
    indexed = index_headings(normalized, KNOWN_HEADINGS)

    sections = {}
    for i, (heading, start) in enumerate(indexed):
        end = indexed[i + 1][1] if i + 1 < len(indexed) else len(normalized)
        section_text = normalized[start + len(heading):end].strip()

        group = FILES[path.name]
        subject = KNOWN_HEADINGS[heading]
        if subject:
            sections[heading] =  Section(
                source=group.source,
                species=group.species,
                stock=group.stock,
                year=group.year,
                file=group.file,
                section=heading,
                text=section_text,
                header=f"""[{group.species} - {subject.title()} - {group.stock} — {heading.title()}]"""
            )
        else:
            print(f"Skipping {heading}")

    return list(sections.values())

def index_headings(text: str, headings: dict) -> list[tuple[str, int]]:
    normalized = re.sub(r'\s+', ' ', text)
    found = []
    for heading, disposition in headings.items():
        idx = normalized.find(heading)
        if idx != -1:
            found.append((heading, idx))
    return sorted(found, key=lambda x: x[1])



# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# TODO(you) #3 — filename → metadata
#   "humpback-caorwa-2021.pdf" -> {source, species, stock, year}
#   Hint: split the stem on '-'. Map the species slug ('humpback' -> 'Humpback Whale')
#   and the region slug ('caorwa' -> 'California/Oregon/Washington') via small dicts.
# ══════════════════════════════════════════════════════════════════════════════
def parse_filename(pdf_path: Path) -> Group:
    if group:= FILES.get(pdf_path.name, None):
        group.file = pdf_path.name
        return group
    raise Exception(f"file {pdf_path.name or '[]'} not found")



# ──────────────────────────────────────────────────────────────────────────────


def extract(pdf_path: Path) -> list[Section]:
    """One SAR → list[Section]  (extract + select + annotate; no chunk/embed yet)."""
    meta = parse_filename(pdf_path)               # TODO(you) #3
    raw = load_text(pdf_path)             # scaffolded
    cleaned = clean(raw)              # scaffolded
    sections: list[Section] = split_into_sections(pdf_path, cleaned)               # TODO(you) #2


    # for heading, body in kept.items():
    #     section = heading.title()                 # normalize "VESSEL STRIKES" -> "Vessel Strikes"
    #     # ── TODO(you) #4: build the contextual header from meta["source"] + section ──
    #     #    e.g.  f'[{meta["source"]} — {section}]'   ->  "[Humpback CA/OR/WA SAR — Vessel Strikes]"
    #     header = None  # <- replace
    #     # ────────────────────────────────────────────────────────────────────────────
    #     sections.append(Section(
    #         source=meta["source"], species=meta["species"], stock=meta["stock"],
    #         year=meta["year"], section=section, text=body, header=header,
    #     ))
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

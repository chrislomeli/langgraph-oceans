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

import dataclasses
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter



import pdfplumber

from config import get_settings
from stores.postgres.embedder import Embedder
from stores.postgres.rag_repo import DocumentRepository

CORPUS = Path.home() / "Source/DATA/oceans/corpus/sar"


# ══════════════════════════════════════════════════════════════════════════════
# TODO(you) #1 — the SAR template + the keep/drop decision
#   List the section headings of a NOAA SAR as they appear in the text (from
#   sar-collection-design.md). Note: MAJOR sections are ALL-CAPS ("STATUS OF STOCK"),
#   subsections are Title-Case ("Vessel Strikes") — include both kinds you care about.
#   Then KEEP_SECTIONS = the subset we keep (the reasoning the card can't hold).
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Group:
    source: str  # full source name,    e.g. "NOAA Fisheries' ... Stock Assessment Reports"
    source_tag: str  # short source tag,    e.g. "SAR"
    species: str  # common name,         e.g. "Humpback Whale"  (= individuals.common_name)
    stock: str  # stock label,         e.g. "California/Oregon/Washington"
    revised: int


@dataclass
class MetaData:
    """One KEPT section of one SAR — the unit the shared chunker will consume."""
    file: str  # source filename,     e.g. "humpback-caorwa-2021.pdf"
    source: str  # full source name,    e.g. "NOAA Fisheries' ... Stock Assessment Reports"
    source_tag: str  # short source tag,    e.g. "SAR"
    species: str  # common name,         e.g. "Humpback Whale"  (= individuals.common_name)
    stock: str  # stock label,         e.g. "California/Oregon/Washington"
    year: int
    section: str  # the heading,         e.g. "Vessel Strikes"
    header: str  # the contextual header (prepended before embedding)


SAR = "NOAA Fisheries' Marine Mammal Stock Assessment Reports"
SHORT_NAME = "SAR"

class CommonName(StrEnum):
    BLUE = "Blue Whale"
    SPERM = "Sperm Whale"
    GRAY = "Gray Whale"
    HUMPBACK = "Humpback Whale"
    FIN = "Fin Whale"
    ORCA = "Killer Whale"


FILES: dict = {
    "blue-whale-enp-2023.pdf": Group(source=SAR, source_tag=SHORT_NAME, species=CommonName.BLUE, stock="Eastern North Pacific", revised=2024),
    "fin-whale-caorwa-2023.pdf": Group(source=SAR, source_tag=SHORT_NAME, species=CommonName.FIN, stock="California Oregon Washington ",
                                       revised=2024),
    "gray-whale-eastern-np-2020.pdf": Group(source=SAR, source_tag=SHORT_NAME, species=CommonName.GRAY, stock="Eastern North Pacific",
                                            revised=2021),
    "humpback-caorwa-2021.pdf": Group(source=SAR, source_tag=SHORT_NAME, species=CommonName.HUMPBACK, stock="California Oregon Washington",
                                      revised=2022),
    "humpback-casmex-dps-2022.pdf": Group(source=SAR, source_tag=SHORT_NAME, species=CommonName.HUMPBACK,
                                          stock="Central America / Southern Mexico - California Oregon Washington",
                                          revised=2023),
    "killer-whale-enp-2024.pdf": Group(source=SAR, source_tag=SHORT_NAME, species=CommonName.ORCA, stock="Eastern Pacific Southern Resident",
                                       revised=2024),
    "killer-whale-southern-resident-2021.pdf": Group(source=SAR, source_tag=SHORT_NAME, species=CommonName.ORCA,
                                                     stock="Eastern North Pacific Southern Resident", revised=2022),
    # "pacific-mmsar-2024-combined.pdf": Group(source=SAR, short_name=SHORT_NAME, species="combined", stock="", revised=0),
    "sperm-whale-caorwa-2023.pdf": Group(source=SAR, source_tag=SHORT_NAME, species=CommonName.SPERM, stock="California Oregon Washington",
                                         revised=2024)
}

KNOWN_HEADINGS: dict = {
    "STOCK DEFINITION AND GEOGRAPHIC RANGE": "STOCK",
    "POPULATION SIZE": "STOCK",
    # "Minimum Population Estimate": "",
    # "Current Population Trend": "",
    "CURRENT AND MAXIMUM NET PRODUCTIVITY RATES": None,
    "POTENTIAL BIOLOGICAL REMOVAL": None,
    "HUMAN-CAUSED MORTALITY AND SERIOUS INJURY": "MORTALITY",  # "Vessel Strikes"
    "Fishery Information": None,
    "REFERENCES": None,
    "Vessel Strikes": "MORTALITY",
    "Other human-caused mortality and serious injury": "MORTALITY",
    "Habitat Concerns": "MORTALITY",
    "STATUS OF STOCK": "STOCK",
}


# ── plumbing: PDF → flat text → cleaned (DONE — read, don't edit unless tuning) ──
def load_text(pdf_path: Path) -> str:
    """pdfplumber: PDF → one flat text string, page by page (no structure survives)."""
    parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            print(f"Reading Page {i}")
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
            continue  # blank line
        if s.isdigit():
            continue  # bare page number
        if re.search(r"\.{4,}", s):
            continue  # TOC dotted-leader line, e.g. "Population Size ......... 7"
        out.append(s)
    return "\n".join(out)


# ────────────────────────────────────────────────────────────────────────────────
def chunk_text(full_text: str, metadata: MetaData, chunk_size: int, overlap : float = 0.2) -> list[Document]:
    chunk_overlap = int(chunk_size * overlap)
    doc = Document(
        page_content=full_text,
        metadata=dataclasses.asdict(metadata)
    )

    # --- STEP 2: SPLITTING ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Order of priority for splitting
    )

    # Split the list of page documents into smaller text chunks
    chunks = text_splitter.split_documents([doc])
    print(f"Successfully generated {len(chunks)} text chunks.")

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# TODO(you) #2 — the heading splitter (THE educational core; "our" heading split)
#   The headings are NOT markup (the PDF lost them), so we supply them. Walk the
#   lines: when a line IS one of KNOWN_HEADINGS, start a new section; otherwise
#   append the line to the current section's body. Return {heading: body} for the
#   sections whose heading is in KEEP_SECTIONS.
#   (You'll discover matching is fiddly — exact match? case-normalize? — that's the
#    learning. Eyeball the checkpoint and tighten it.)
# ══════════════════════════════════════════════════════════════════════════════
def split_into_sections(path: Path, text: str, group: Group) -> list[Document]:
    # normalized = re.sub(r'\s+', ' ', text)
    indexed = index_headings(text, KNOWN_HEADINGS)

    sections = []
    for ndx, (heading, start) in enumerate(indexed):
        end = indexed[ndx + 1][1] if ndx + 1 < len(indexed) else len(text)
        section_text = text[start + len(heading):end].strip()

        subject = KNOWN_HEADINGS[heading]

        if subject:
            metadata = MetaData(
                source=group.source,
                source_tag=group.source_tag,
                species=str(group.species),
                stock=group.stock,
                year=group.revised,
                file=path.name,
                section=heading,
                header=f"""[{group.species} - {subject.title()} - {group.stock} — {heading.title()}]"""
            )

            # page_content stays the BARE body (what gets cited + stored in text).
            # The header rides in metadata["header"]; the repo prepends it only for
            # the embedding call, so the vector carries species+topic but text does not.
            chunks = chunk_text(full_text=section_text, metadata=metadata, chunk_size=512)
            sections.extend(chunks)

        else:
            print(f"Skipping {heading}")

    for i, doc in enumerate(sections):
        ndx = i + 1
        cid = f"{path.stem}.{ndx}"
        doc.metadata["chunk_index"] = ndx  # per-chunk
        doc.metadata["chunk_id"] = cid # per-chunk, deterministic
        doc.id = cid

    return sections

def index_headings(text: str, headings: dict) -> list[tuple[str, int]]:
    found = []
    for heading, disposition in headings.items():
        pattern = re.compile(r"^\s*" + r"\s+".join(heading.split()), re.MULTILINE | re.IGNORECASE)
        match = pattern.search(text)
        if match:
            found.append((heading, match.start()))

    return sorted(found, key=lambda x: x[1])

# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# TODO(you) #3 — filename → metadata
#   "humpback-caorwa-2021.pdf" -> {source, species, stock, year}
#   Hint: split the stem on '-'. Map the species slug ('humpback' -> 'Humpback Whale')
#   and the region slug ('caorwa' -> 'California/Oregon/Washington') via small dicts.
# ══════════════════════════════════════════════════════════════════════════════
def parse_filename(pdf_path: Path) -> Group:
    if group := FILES.get(pdf_path.name, None):
        # group.file = pdf_path.name
        return group
    raise Exception(f"file {pdf_path.name or '[]'} not found")


# ──────────────────────────────────────────────────────────────────────────────

def extract(repo: DocumentRepository, pdf_path: Path) -> list[Document]:
    """One SAR → list[Section]  (extract + select + annotate; no chunk/embed yet)."""
    group = parse_filename(pdf_path)  # TODO(you) #3
    raw = load_text(pdf_path)  # scaffolded
    clean_text = clean(raw)  # scaffolded
    documents: list[Document] = split_into_sections(path=pdf_path, text=clean_text, group = group)  # TODO(you) #2
    repo.save_chunks(documents)
    return documents

if __name__ == "__main__":
    # CHECKPOINT — all docs, print only. Two verifications beyond a section count:
    #   (1) COVERAGE DIFF: which KNOWN_HEADINGS each doc matched vs missed. A non-empty
    #       MISSING set on one doc = a reworded/uncased heading the exact match dropped.
    #   (2) BOUNDARY PEEK: for each kept chunk, the anchor text + head/tail of the body,
    #       so you can see boundaries land on real heading lines and bodies don't bleed.
    settings = get_settings()
    embedder = Embedder(settings.openai_api_key.get_secret_value())
    repo = DocumentRepository( embedder=embedder )

    for name in FILES:
        print(f"""\n=========================\n{name}\n================================""")
        pdf = CORPUS / name
        cleaned = clean(load_text(pdf))

        # (1) coverage diff — KNOWN_HEADINGS is the SAR template = ground truth
        print("\nIndexing...")
        indexed = index_headings(cleaned, KNOWN_HEADINGS)
        found = {h for h, _ in indexed}
        missing = [h for h in KNOWN_HEADINGS if h not in found]

        print("\nProcessing...")
        secs: list[Document] = extract(repo, pdf)
        kept = sorted({s.metadata["section"] for s in secs})
        print(f"\n{'='*88}\n{name}: {len(secs)} chunks across {len(kept)} sections "
              f"| matched {len(found)}/{len(KNOWN_HEADINGS)} headings")
        print(f"  MISSING: {missing or 'none'}")

        # (2) boundary peek — head/tail of each chunk's body
        for s in secs:
            body = s.page_content
            print(f"  --- {s.id}  [{s.metadata['section']}]  ({len(body)} chars)")
            print(f"        head: {body[:80]!r}")
            print(f"        tail: {body[-80:]!r}")

#!/usr/bin/env python3
"""
HR Attrition Intelligence - Layer 1 ingestion
=============================================
Downloads quarterly fact-sheet PDFs (TCS, Infosys), extracts attrition and
headcount metrics, merges a manual CSV (Wipro / HCLTech / Tech Mahindra),
and optionally loads everything into PostgreSQL.

Usage
-----
    pip install -r requirements.txt

    python ingest_attrition.py download            # fetch PDFs into data/raw/
    python ingest_attrition.py extract             # PDFs -> data/processed/company_attrition.csv
    python ingest_attrition.py load                # CSV -> Postgres (needs DATABASE_URL)
    python ingest_attrition.py all                 # download + extract

    python ingest_attrition.py download --company tcs --start 2022 --end 2027

Notes
-----
* FY is the Indian fiscal year; --start/--end are the calendar year in which
  the FY ENDS (FY26 = Apr 2025 - Mar 2026 -> 2026).
* Extraction is regex over PDF text. ALWAYS spot-check rows whose status is
  'needs_review' and a random sample of 'ok' rows against the source PDF.
* Definitions changed over time (LTM vs voluntary LTM). The `definition`
  column keeps this visible - do not mix them blindly in analysis.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pdfplumber
import requests

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MANUAL_CSV = ROOT / "data" / "manual" / "manual_attrition.csv"
OUT_CSV = PROC_DIR / "company_attrition.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio-project; personal use)"}
log = logging.getLogger("attrition")

COLUMNS = [
    "company", "fiscal_year", "quarter", "period_end", "attrition_pct",
    "definition", "headcount", "gross_adds", "net_adds", "attrition_count",
    "source_url", "source_file", "status",
]


# --------------------------------------------------------------------------
# Calendar helpers
# --------------------------------------------------------------------------
def quarter_end(fy_end: int, q: int) -> date:
    """Indian FY: Q1=Apr-Jun, Q2=Jul-Sep, Q3=Oct-Dec, Q4=Jan-Mar."""
    return {
        1: date(fy_end - 1, 6, 30),
        2: date(fy_end - 1, 9, 30),
        3: date(fy_end - 1, 12, 31),
        4: date(fy_end, 3, 31),
    }[q]


def fy_label(fy_end: int) -> str:
    return f"FY{fy_end % 100:02d}"


# --------------------------------------------------------------------------
# URL builders (patterns verified against published links)
# --------------------------------------------------------------------------
def tcs_urls(fy_end: int, q: int) -> list[str]:
    fy = f"{fy_end - 1}-{fy_end % 100:02d}"  # e.g. 2025-26
    base = (
        "https://www.tcs.com/content/dam/tcs/investor-relations/"
        f"financial-statements/{fy}/q{q}/Presentations/"
    )
    names = [
        f"Q{q} {fy} Fact Sheet.pdf",  # FY22, FY24 onwards
        f"Q{q} FY {fy} Fact sheet.pdf",  # FY23 variant
    ]
    return [quote(base + n, safe=":/") for n in names]


def infosys_urls(fy_end: int, q: int) -> list[str]:
    fy = f"{fy_end - 1}-{fy_end}"  # e.g. 2010-2011
    base = (
        "https://www.infosys.com/investors/reports-filings/"
        f"quarterly-results/{fy}/q{q}/documents/"
    )
    # filename varies by year
    return [base + n for n in ("fact-sheet.pdf", "factsheet.pdf", "factsheet1.pdf")]


# company -> (url builder, default start FY-end year, default end FY-end year)
SOURCES = {
    "tcs": (tcs_urls, 2020, 2027),
    "infosys": (infosys_urls, 2004, 2027),  # same URL pattern confirmed for FY26
}


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def fetch(url: str, dest: Path, session: requests.Session, retries: int=3) -> bool:
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=45)
        except requests.RequestException as exc:
            log.warning("network error on %s: %s", url, exc)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            return True
        if r.status_code == 404:
            return False
        if r.status_code == 403:
            log.warning("403 (blocked?) %s", url)
            return False
        time.sleep(2 ** attempt)
    return False


def download_company(company: str, start: int, end: int, delay: float) -> None:
    builder, d_start, d_end = SOURCES[company]
    start = start or d_start
    end = end or d_end
    out_dir = RAW_DIR / company
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    session = requests.Session()
    got = missing = 0
    for fy_end in range(start, end + 1):
        for q in range(1, 5):
            name = f"{company}_{fy_label(fy_end)}_Q{q}.pdf"
            dest = out_dir / name
            if dest.exists():
                continue
            for url in builder(fy_end, q):
                if fetch(url, dest, session):
                    manifest[name] = url
                    got += 1
                    log.info("downloaded %s", name)
                    break
            else:
                missing += 1
                log.debug("not found: %s %s Q%d", company, fy_label(fy_end), q)
            time.sleep(delay)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("%s: %d new PDFs, %d not found", company, got, missing)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def iter_pdf_texts(path: Path):
    """Yield text from several PDF readers, best first.

    Some sheets come out garbled with pdfplumber (letters spaced out, numbers
    scattered), so PyMuPDF is used as a fallback if it is installed.
    """
    try:
        with pdfplumber.open(path) as pdf:
            yield "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception as exc:
        log.warning("pdfplumber failed on %s: %s", path.name, exc)
    try:
        try:
            import pymupdf as fitz  # PyMuPDF (new import name)
        except ImportError:
            import fitz  # older PyMuPDF
    except ImportError:
        log.debug("PyMuPDF not installed; no fallback reader")
        return
    try:
        with fitz.open(path) as doc:
            yield "\n".join(page.get_text() for page in doc)
    except Exception as exc:
        log.warning("PyMuPDF failed on %s: %s", path.name, exc)


def to_int(s: str | None) -> int | None:
    return int(s.replace(",", "")) if s else None


PCT = r"(?P<pct>\d{1,2}(?:\.\d+)?)"
TCS_ATTR_PATTERNS = [
    # "Voluntary LTM attrition at 13.3% in IT Services"
    rf"(?:Voluntary\s+)?LTM\s+attrition[^%\n]{{0,40}}?(?:at|to)\s+{PCT}\s*%",
    # "Voluntary Attrition*: • 13.3% (LTM), IT Services"
    rf"(?:Voluntary\s+)?Attrition\*?\s*:?\s*[•\-\u2022\s]*{PCT}\s*%?\s*\(LTM\)",
    # "• 13.3% LTM Attrition* – IT Services"
    rf"{PCT}\s*%\s*LTM\s+Attrition",
]


def extract_tcs(text: str) -> dict:
    out = {"attrition_pct": None, "definition": None, "headcount": None}
    for pat in TCS_ATTR_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            out["attrition_pct"] = float(m.group("pct"))
            out["definition"] = (
                "voluntary_ltm" if "voluntary" in m.group(0).lower() else "ltm_unspecified"
            )
            break
    m = re.search(r"Closing\s+headcount\s*:\s*([\d,]+)", text, re.I)
    out["headcount"] = to_int(m.group(1)) if m else None
    return out


def extract_infosys(text: str) -> dict:
    out = {"attrition_pct": None, "definition": "total_ltm", "headcount": None,
           "gross_adds": None, "net_adds": None, "attrition_count": None}

    # Layout A (FY04-FY15-ish): label and values on the same line.
    m = re.search(r"Attrition\s*%?\s*\(LTM\)\*?\s+([\d.]+)", text, re.I)
    if m:
        out["attrition_pct"] = float(m.group(1))
    m = re.search(r"Gross\s+addition\s+([\d,]+)", text, re.I)
    out["gross_adds"] = to_int(m.group(1)) if m else None
    m = re.search(r"Net\s+addition\s+([\d,]+)", text, re.I)
    out["net_adds"] = to_int(m.group(1)) if m else None
    m = re.search(r"(?m)^Attrition\s+([\d,]+)\b", text)
    out["attrition_count"] = to_int(m.group(1)) if m else None
    m = re.search(r"Total\s+employees\s+([\d,]+)", text, re.I)
    out["headcount"] = to_int(m.group(1)) if m else None

    # Layout C (FY16-FY20-ish): annualized quarterly attrition, NOT LTM.
    #   "Attrition % (Annualized Consolidated) 23.4% 20.4% 23.0%"
    if out["attrition_pct"] is None:
        for label, defn in (("Consolidated", "annualized_consolidated"),
                            ("Standalone", "annualized_standalone")):
            m = re.search(rf"Attrition\s*%\s*\(Annualized\s+{label}\)\s+([\d.]+)", text, re.I)
            if m:
                out["attrition_pct"] = float(m.group(1))
                out["definition"] = defn
                break

    # Layout D (FY21): voluntary, annualized, IT services only - a different basis
    if out["attrition_pct"] is None:
        m = re.search(r"Voluntary\s+Attrition\s*%\s*\(Annualized\s*-\s*IT\s+Services\)\s+([\d.]+)", text, re.I)
        if m:
            out["attrition_pct"] = float(m.group(1))
            out["definition"] = "annualized_voluntary_it"

    # Layout B (recent sheets): labels and values sit in separate blocks.
    # The first percentage after "Employee Metrics" (before "Cash Flow") is the
    # current-quarter attrition; the first big number is total employees.
    if out["attrition_pct"] is None and re.search(r"Attrition\s*%\s*\(LTM", text, re.I):
        sec = re.search(r"Employee Metrics(.*?)(?:Cash Flow|$)", text, re.S | re.I)
        if sec:
            body = sec.group(1)
            m = re.search(r"(\d{1,2}\.\d)\s*%", body)
            if m:
                out["attrition_pct"] = float(m.group(1))
            if out["headcount"] is None:
                m = re.search(r"(\d{1,3}(?:,\d{3})+)", body)
                out["headcount"] = to_int(m.group(1)) if m else None

    if out["definition"] == "total_ltm" and re.search(r"Voluntary\s+Attrition", text, re.I):
        out["definition"] = "voluntary_ltm"
    return out


EXTRACTORS = {"tcs": extract_tcs, "infosys": extract_infosys}


def extract_company(company: str) -> list[dict]:
    rows = []
    folder = RAW_DIR / company
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for pdf in sorted(folder.glob("*.pdf")):
        m = re.search(r"_FY(\d{2})_Q(\d)\.pdf$", pdf.name)
        if not m:
            continue
        fy_end, q = 2000 + int(m.group(1)), int(m.group(2))
        data: dict = {}
        for text in iter_pdf_texts(pdf):
            try:
                cand = EXTRACTORS[company](text)
            except Exception as exc:
                log.error("failed to parse %s: %s", pdf.name, exc)
                continue
            if not data:
                data = cand
            if cand.get("attrition_pct") is not None:
                data = cand
                break
        pct = data.get("attrition_pct")
        row = {
            "company": company.upper() if company == "tcs" else company.capitalize(),
            "fiscal_year": fy_label(fy_end),
            "quarter": q,
            "period_end": quarter_end(fy_end, q).isoformat(),
            "attrition_pct": pct,
            "definition": data.get("definition"),
            "headcount": data.get("headcount"),
            "gross_adds": data.get("gross_adds"),
            "net_adds": data.get("net_adds"),
            "attrition_count": data.get("attrition_count"),
            "source_url": manifest.get(pdf.name),
            "source_file": str(pdf.relative_to(ROOT)),
            "status": "ok" if pct is not None and 0 < pct < 60 else "needs_review",
        }
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Manual data (Wipro, HCLTech, Tech Mahindra) - fill from investor pages
# --------------------------------------------------------------------------
def ensure_manual_template() -> None:
    if MANUAL_CSV.exists():
        return
    MANUAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    seed = [
        # Seeded from press reports - VERIFY against Wipro's own fact sheet
        ["Wipro", "FY26", 1, "2025-06-30", 15.1, "voluntary_ltm", "", "", "", "",
         "https://www.businesstoday.in/amp/technology/story/wipro-adds-135-employees-in-q4-attrition-eases-to-14-526016-2026-04-16",
         "manual", "manual_verify"],
        ["Wipro", "FY26", 3, "2025-12-31", 14.2, "voluntary_ltm", "", "", "", "",
         "https://www.businesstoday.in/amp/technology/story/wipro-adds-135-employees-in-q4-attrition-eases-to-14-526016-2026-04-16",
         "manual", "manual_verify"],
        ["Wipro", "FY26", 4, "2026-03-31", 13.8, "voluntary_ltm", "", "", "", "",
         "https://www.businesstoday.in/amp/technology/story/wipro-adds-135-employees-in-q4-attrition-eases-to-14-526016-2026-04-16",
         "manual", "manual_verify"],
    ]
    pd.DataFrame(seed, columns=COLUMNS).to_csv(MANUAL_CSV, index=False)
    log.info("created manual template: %s", MANUAL_CSV)


def build_dataset(companies: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for c in companies:
        if (RAW_DIR / c).exists():
            rows += extract_company(c)
    auto = pd.DataFrame(rows, columns=COLUMNS)

    ensure_manual_template()
    manual = pd.read_csv(MANUAL_CSV)
    manual = manual.reindex(columns=COLUMNS)

    df = pd.concat([auto, manual], ignore_index=True)
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.date
    df = (
        df.sort_values(["company", "period_end"])
        .drop_duplicates(["company", "period_end", "definition"], keep="first")
        .reset_index(drop=True)
    )
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    bad = (df["status"] == "needs_review").sum()
    log.info("saved %d rows to %s (%d need review)", len(df), OUT_CSV, bad)
    return df


# --------------------------------------------------------------------------
# Postgres
# --------------------------------------------------------------------------
def load_postgres() -> None:
    from sqlalchemy import create_engine, text

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit(
            "Set DATABASE_URL, e.g. postgresql+psycopg2://user:pass@localhost:5432/hr_attrition"
        )
    engine = create_engine(url)
    df = pd.read_csv(OUT_CSV)
    with engine.begin() as conn:
        # keep the table (views depend on it): empty it, then append
        if conn.execute(text("SELECT to_regclass('raw.company_attrition')")).scalar():
            conn.execute(text("TRUNCATE raw.company_attrition"))
    df.to_sql("company_attrition", engine, schema="raw", if_exists="append", index=False)
    log.info("loaded %d rows into raw.company_attrition", len(df))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["download", "extract", "load", "all"])
    ap.add_argument("--company", choices=[*SOURCES, "all"], default="all")
    ap.add_argument("--start", type=int, default=0, help="first FY-end year (e.g. 2022)")
    ap.add_argument("--end", type=int, default=0, help="last FY-end year (e.g. 2027)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    companies = list(SOURCES) if args.company == "all" else [args.company]

    if args.command in ("download", "all"):
        for c in companies:
            download_company(c, args.start, args.end, args.delay)
    if args.command in ("extract", "all"):
        df = build_dataset(companies)
        print(df[["company", "fiscal_year", "quarter", "attrition_pct", "definition", "status"]].tail(15))
    if args.command == "load":
        load_postgres()


if __name__ == "__main__":
    main()

"""Load the RBI "DLAs deployed by Regulated Entities" list (verified class).

The list is published at rbi.org.in -> Citizen's Corner -> "DLA's deployed by
Regulated Entities", which opens a SAP BusinessObjects OpenDocument report on
data.rbi.org.in. That viewer is JavaScript-driven and has proven unreliable for
automated access (HTTP 500 / login redirect), so the supported path is:

  1. Open the link in a browser, export the report (Excel or CSV).
  2. Save it as data/raw/rbi_dla_manual_<YYYY-MM-DD>.<xlsx|xls|csv|html>.
  3. Run: python -m loanshield.collect.rbi_directory

The newest manual file is normalised to data/raw/rbi_dla_<date>.csv with
columns dla_name, re_name, re_type (+ play_url / package_id if present).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from loanshield.common import get_logger, path

log = get_logger("rbi_directory")
PKG_RE = re.compile(r"id=([A-Za-z0-9_.]+)")

# substrings (lowercase) used to recognise the report's column headers
COLUMN_HINTS = {
    "dla_name": ["dla name", "name of dla", "app name", "name of the app", "dla"],
    "re_name": ["re name", "name of re", "regulated entity", "name of the re", "entity name"],
    "re_type": ["re type", "type of re", "category", "entity type"],
    "play_url": ["play store", "link", "url", "android"],
}


def read_any(f: Path) -> pd.DataFrame:
    suf = f.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(f)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(f)
    if suf in (".html", ".htm"):
        return max(pd.read_html(f), key=len)
    raise ValueError(f"unsupported file type: {f}")


def find_header(df: pd.DataFrame) -> pd.DataFrame:
    """BO exports often have title rows above the real header; locate it."""
    def score(cols) -> int:
        cl = [str(c).lower() for c in cols]
        return sum(any(h in c for c in cl for h in hints) for hints in COLUMN_HINTS.values())

    if score(df.columns) >= 2:
        return df
    for i in range(min(20, len(df))):
        if score(df.iloc[i].values) >= 2:
            out = df.iloc[i + 1:].copy()
            out.columns = [str(c).strip() for c in df.iloc[i].values]
            return out.reset_index(drop=True)
    raise ValueError("could not locate header row; inspect the file and extend COLUMN_HINTS")


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping, used = {}, set()
    for target, hints in COLUMN_HINTS.items():
        for h in hints:  # hints are ordered most- to least-specific
            hit = next((c for c in df.columns if c not in used and h in str(c).lower()), None)
            if hit is not None:
                mapping[hit] = target
                used.add(hit)
                break
    log.info("column mapping: %s", mapping)
    out = df.rename(columns=mapping)
    missing = {"dla_name", "re_name"} - set(out.columns)
    if missing:
        raise ValueError(f"missing required columns {missing}; got {list(df.columns)}")
    return out


def norm_re_type(s: str) -> str:
    s = str(s).lower()
    if "bank" in s:
        return "bank"
    if "housing" in s or "hfc" in s:
        return "HFC"
    if "nbfc" in s or "non-banking" in s or "non banking" in s:
        return "NBFC"
    return "other" if s and s != "nan" else ""


def main() -> None:
    raw = path("raw")
    manual = sorted(raw.glob("rbi_dla_manual_*.*"))
    if not manual:
        log.error("No manual RBI file found. Download the DLA list (see module docstring) and save "
                  "it as data/raw/rbi_dla_manual_<YYYY-MM-DD>.xlsx. Not generating any data.")
        sys.exit(1)
    f = manual[-1]
    snap = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
    snap = snap.group(1) if snap else "undated"
    df = map_columns(find_header(read_any(f)))
    df = df.dropna(subset=["dla_name"])
    df["dla_name"] = df["dla_name"].astype(str).str.strip()
    df["re_name"] = df["re_name"].astype(str).str.strip()
    df["re_type"] = df["re_type"].map(norm_re_type) if "re_type" in df else ""
    if "play_url" in df:
        df["package_id"] = df["play_url"].astype(str).str.extract(PKG_RE, expand=False)
    keep = [c for c in ["dla_name", "re_name", "re_type", "play_url", "package_id"] if c in df]
    out = raw / f"rbi_dla_{snap}.csv"
    df[keep].drop_duplicates().to_csv(out, index=False)
    log.info("%d DLA rows (%d unique REs) from %s -> %s", len(df), df["re_name"].nunique(), f.name, out)
    log.info("re_type counts: %s", df["re_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()

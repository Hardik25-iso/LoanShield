"""Load the RBI "DLAs deployed by Regulated Entities" list (verified class).

The list is published at rbi.org.in -> Citizen's Corner -> "DLA's deployed by
Regulated Entities", which opens a SAP BusinessObjects OpenDocument report on
data.rbi.org.in. That viewer is JavaScript-driven and has proven unreliable for
plain HTTP access (HTTP 500 / logon redirect). Two supported inputs:

  a. data/raw/rbi_dla_extract_<YYYY-MM-DD>.json - rows read from the rendered
     WebI table in a browser session (all pages; grievance-officer contact
     columns dropped). Format: {"header": [...], "rows": [[...], ...]}.
  b. data/raw/rbi_dla_manual_<YYYY-MM-DD>.<xlsx|xls|csv|html> - a manual export.

Run: python -m loanshield.collect.rbi_directory

The newest input is normalised to data/raw/rbi_dla_<date>.csv with columns
dla_name, re_name, re_type, dla_owner, re_website, available_on, play_url, package_id.
"""
from __future__ import annotations

import json
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
    "play_url": ["link to dla", "play store", "link", "android"],
    "dla_owner": ["owner"],
    "re_website": ["website of re"],
    "available_on": ["available on"],
}


def read_any(f: Path) -> pd.DataFrame:
    suf = f.suffix.lower()
    if suf == ".json":
        d = json.loads(f.read_text(encoding="utf-8"))
        return pd.DataFrame(d["rows"], columns=d["header"])
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
    if "housing" in s or "hfc" in s:
        return "HFC"
    if "bank" in s:
        return "bank"
    if "nbfc" in s or "non-banking" in s or "non banking" in s:
        return "NBFC"
    return "other" if s and s != "nan" else ""


def main() -> None:
    raw = path("raw")
    inputs = list(raw.glob("rbi_dla_manual_*.*")) + list(raw.glob("rbi_dla_extract_*.json"))
    if not inputs:
        log.error("No RBI input found (see module docstring). Not generating any data.")
        sys.exit(1)
    f = max(inputs, key=lambda x: re.search(r"\d{4}-\d{2}-\d{2}", x.name).group(0)
            if re.search(r"\d{4}-\d{2}-\d{2}", x.name) else "")
    snap = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
    snap = snap.group(1) if snap else "undated"
    df = map_columns(find_header(read_any(f)))
    df = df.dropna(subset=["dla_name"])
    df["dla_name"] = df["dla_name"].astype(str).str.strip()
    df["re_name"] = df["re_name"].astype(str).str.strip()
    df["re_type"] = df["re_type"].map(norm_re_type) if "re_type" in df else ""
    if "play_url" in df:
        df["package_id"] = df["play_url"].astype(str).str.extract(PKG_RE, expand=False)
    keep = [c for c in ["dla_name", "re_name", "re_type", "dla_owner", "re_website",
                        "available_on", "play_url", "package_id"] if c in df]
    out = raw / f"rbi_dla_{snap}.csv"
    df[keep].drop_duplicates().to_csv(out, index=False)
    log.info("%d DLA rows (%d unique REs) from %s -> %s", len(df), df["re_name"].nunique(), f.name, out)
    log.info("re_type counts: %s", df["re_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()

"""Listing metadata, developer and listing-text features (one row per app).

Leakage rule (PLAN.md section 5): nothing here reads the RBI directory or the
match scores. `mentions_partner` is a regex on the app's own description.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from loanshield.common import cfg

INSTALL_BANDS = [0, 1, 5, 10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000,
                 1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000,
                 1_000_000_000, 5_000_000_000]
FREE_MAIL = {"gmail.com", "yahoo.com", "yahoo.in", "yahoo.co.in", "outlook.com", "hotmail.com",
             "rediffmail.com", "protonmail.com", "proton.me", "icloud.com", "aol.com", "zoho.com",
             "live.com", "mail.com", "yandex.com", "gmx.com", "qq.com", "163.com"}

RE_PARTNER = re.compile(r"\bnbfc\b|lending partner|rbi[- ]registered|registered with (the )?rbi|"
                        r"in partnership with|partnered with|regulated entit|\bbank (ltd|limited)\b", re.I)
RE_APR = re.compile(r"\bapr\b|annual percentage rate|\d+(\.\d+)?\s?%\s*(p\.?a\.?|per annum|annual)|"
                    r"interest rate[^.]{0,40}\d+(\.\d+)?\s?%", re.I)
RE_TENURE = re.compile(r"\b\d+\s*(?:-|to)?\s*\d*\s*(days?|months?|years?)\b[^.]{0,30}"
                       r"(tenure|repay|loan term)|tenure[^.]{0,40}\d+\s*(days?|months?|years?)", re.I)
RE_KFS = re.compile(r"\bkfs\b|key fact statement|grievance (redressal )?officer|nodal officer", re.I)
URGENCY = [r"\binstant(ly)?\b", r"\b\d+\s*min(ute)?s?\b", r"no documents?", r"without documents?",
           r"no cibil", r"without cibil", r"low cibil", r"minimal documentation", r"paperless",
           r"no credit (score|check|history)", r"in minutes", r"quick approval", r"100% approval",
           r"guaranteed", r"\burgent\b"]
RE_URGENCY = [re.compile(p, re.I) for p in URGENCY]


def _date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="mixed", errors="coerce")


def _domain(url) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    host = urlparse(url if "//" in url else f"//{url}").netloc.lower().split(":")[0]
    parts = [p for p in host.split(".") if p and p != "www"]
    # registered domain: last two labels, or last three for co.in-style suffixes
    keep = 3 if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "gov", "ac"} else 2
    return ".".join(parts[-keep:])


def install_band(n) -> float:
    if pd.isna(n):
        return np.nan
    return float(np.searchsorted(INSTALL_BANDS, n, side="right") - 1)


def metadata_features(apps: pd.DataFrame) -> pd.DataFrame:
    snap = pd.Timestamp(cfg()["snapshot_date"])
    released = _date(apps["released"])
    updated = pd.to_datetime(apps["updated"], unit="s", errors="coerce")
    hist = apps[[f"hist_{i}" for i in range(1, 6)]].astype(float)
    tot = hist.sum(axis=1).replace(0, np.nan)
    pp_dom = apps["privacyPolicy"].map(_domain)
    web_dom = apps["developerWebsite"].map(_domain)
    return pd.DataFrame({
        "app_id": apps["app_id"],
        "log_min_installs": np.log10(apps["minInstalls"].astype(float) + 1),
        "install_band": apps["minInstalls"].map(install_band),
        "score": apps["score"],
        "has_score": apps["score"].notna().astype(int),
        "log_ratings": np.log10(apps["ratings"].fillna(0).astype(float) + 1),
        "share_1star": hist["hist_1"] / tot,
        "share_5star": hist["hist_5"] / tot,
        "polarisation": (hist["hist_1"] + hist["hist_5"]) / tot,
        "app_age_days": (snap - released).dt.days,
        "days_since_update": (snap - updated).dt.days,
        "update_missing": updated.isna().astype(int),
        "contains_ads": apps["containsAds"].astype(float),
        "has_privacy_policy": (pp_dom != "").astype(int),
        "pp_domain_matches_site": ((pp_dom != "") & (pp_dom == web_dom)).astype(int),
    })


def developer_features(apps: pd.DataFrame) -> pd.DataFrame:
    dev = apps["developerId"].fillna(apps["developer"])
    email_dom = apps["developerEmail"].fillna("").str.lower().str.split("@").str[-1]
    near_dup = pd.Series(0, index=apps.index)
    for _, idx in apps.groupby(dev).groups.items():
        titles = apps.loc[idx, "title"].fillna("").str.lower()
        for i in idx:
            near_dup[i] = sum(fuzz.token_sort_ratio(titles[i], titles[j]) >= 85 for j in idx if j != i)
    return pd.DataFrame({
        "app_id": apps["app_id"],
        "dev_app_count": dev.map(dev.value_counts()),
        "dev_email_free": email_dom.isin(FREE_MAIL).astype(int),
        "dev_email_missing": (email_dom == "").astype(int),
        "dev_email_matches_site": ((email_dom != "") & (email_dom.map(_domain) ==
                                   apps["developerWebsite"].map(_domain))).astype(int),
        "dev_has_website": apps["developerWebsite"].notna().astype(int),
        "dev_near_dup_titles": near_dup,
    })


def _flag(text: pd.Series, rx: re.Pattern) -> pd.Series:
    return text.map(lambda t: int(bool(rx.search(t))))


def listing_text_features(apps: pd.DataFrame) -> pd.DataFrame:
    desc = apps["description"].fillna("")
    full = apps["title"].fillna("") + " " + apps["summary"].fillna("") + " " + desc
    return pd.DataFrame({
        "app_id": apps["app_id"],
        "mentions_partner": _flag(full, RE_PARTNER),
        "states_apr": _flag(full, RE_APR),
        "states_tenure": _flag(full, RE_TENURE),
        "mentions_kfs_grievance": _flag(full, RE_KFS),
        "urgency_terms": full.map(lambda t: sum(bool(r.search(t)) for r in RE_URGENCY)),
        "log_desc_len": np.log10(desc.str.len() + 1),
    })

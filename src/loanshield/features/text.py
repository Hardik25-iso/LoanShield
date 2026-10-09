"""Review-text cleaning and lexicon features (English + Hinglish).

clean_review(): lowercase, strip URLs / phone numbers / emojis, normalise common
Hinglish spellings, collapse whitespace. Reviews are deduplicated per app on the
cleaned text.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_PHONE = re.compile(r"(?<!\d)(\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)|(?<!\d)\d{10,}(?!\d)")
RE_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF"
                      "\U00002190-\U000021FF\U00002B00-\U00002BFF️‍]+")
RE_SPACE = re.compile(r"\s+")

# common Hinglish spelling variants -> one canonical form (word-boundary replace)
HINGLISH = {
    "paisa": ["paise", "pese", "pesa", "paisaa", "paiso"],
    "nahi": ["nahin", "nhi", "nai", "nahee", "nahi"],
    "bahut": ["bohot", "bahot", "bht", "bhut", "bohut"],
    "loan": ["lon", "laon", "lone"],
    "fraud": ["froud", "frod", "fraudd", "fraaud"],
    "dhoka": ["dhokha", "dhokebaaz", "dhokebaz", "dhokhebaz", "dokha"],
    "chor": ["chorr", "choor"],
    "gaali": ["gali", "galli", "gaaliyan", "galiyan", "galiya"],
    "dhamki": ["dhamkee", "dhamakee", "dhamakiyan"],
    "paise_kaat": ["paise kaat", "paisa kaat", "kaat liya", "kat liya", "kaat lia"],
    "rishtedar": ["rishtedaar", "relative", "relatives"],
    "ghar_wale": ["ghar wale", "gharwalo", "ghar walo", "family wale"],
    "kya": ["kia", "kyaa"],
    "bekar": ["bakwas", "bekaar", "bakwaas"],
}
_HMAP = {v: k for k, vs in HINGLISH.items() for v in vs}
_HRE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _HMAP), key=len, reverse=True)) + r")\b")

LEXICONS = {
    # "contact" alone is mostly "contact customer care"; require plural or an access context
    "harass": [r"\bcontacts\b", r"(access|read|upload|messag|call)\w* (to )?(my |all |your )?contact",
               r"call(ing|ed|s)? (my )?(family|relatives?|friends|contacts|parents)",
               r"\bfamily\b", r"\brishtedar\b", r"\bghar_wale\b", r"\babus\w*", r"\bthreat\w*",
               r"\bblackmail\w*", r"\bmorph\w*", r"\bharass\w*", r"\bgaali\b", r"\bdhamki\b",
               r"\bviral\b", r"\bdefam\w*", r"\btortur\w*", r"\bhumiliat\w*", r"\bnude\b"],
    "fee": [r"processing fees?", r"\bdeduct\w*", r"extra charges?", r"hidden (charges?|fees?)",
            r"\bgst\b", r"less amount", r"\bpaise_kaat\b", r"\bcut\b", r"high interest",
            r"interest (is|was) (too )?high", r"\bkam paisa\b", r"\b(30|35|40|50)\s?%"],
    "fraud": [r"\bfraud\w*", r"\bscam\w*", r"\bfake\b", r"\bcheat\w*", r"\bloot\w*", r"\bchor\b",
              r"\bdhoka\b", r"\billegal\b", r"chinese app", r"\brbi\b.{0,20}(complain|report)",
              r"\bcyber ?crime\b", r"\bpolice\b", r"\bfir\b"],
}
LEX_RE = {k: re.compile("|".join(v)) for k, v in LEXICONS.items()}


def clean_review(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = RE_URL.sub(" ", t)
    t = RE_PHONE.sub(" ", t)
    t = RE_EMOJI.sub(" ", t)
    t = _HRE.sub(lambda m: _HMAP[m.group(1)], t)
    return RE_SPACE.sub(" ", t).strip()


def clean_reviews(rv: pd.DataFrame) -> pd.DataFrame:
    """One row per (app, unique cleaned review); empty texts dropped."""
    out = rv.copy()
    out["text"] = out["content"].map(clean_review)
    out = out[out["text"].str.len() > 0]
    out["_key"] = out["text"].str.replace(r"[^\w ]+", "", regex=True).str.strip()
    out = out.sort_values("at").drop_duplicates(["app_id", "_key"], keep="first")
    for k, r in LEX_RE.items():
        out[f"lex_{k}"] = out["text"].map(lambda t, r=r: int(bool(r.search(t))))
    return out[["app_id", "lang", "text", "score", "at", "thumbsUpCount", "appVersion"]
               + [f"lex_{k}" for k in LEX_RE]].reset_index(drop=True)


def review_features(rc: pd.DataFrame, app_ids: pd.Series) -> pd.DataFrame:
    """Aggregate cleaned reviews to app level. Apps with no reviews get NaN + has_reviews=0."""
    g = rc.groupby("app_id")
    agg = pd.DataFrame({
        "rev_count": g.size(),
        "rev_mean_score": g["score"].mean(),
        "rev_share_1star": g["score"].apply(lambda s: (s == 1).mean()),
        **{f"rev_share_{k}": g[f"lex_{k}"].mean() for k in LEX_RE},
        "rev_mean_len": g["text"].apply(lambda s: s.str.len().mean()),
    })
    out = agg.reindex(app_ids.values)
    out["has_reviews"] = out["rev_count"].notna().astype(int)
    out["log_rev_count"] = np.log10(out["rev_count"].fillna(0) + 1)
    return out.drop(columns="rev_count").rename_axis("app_id").reset_index()

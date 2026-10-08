"""Match Play apps to the RBI DLA directory and decide whether an app offers credit.

Matching evidence per app (strongest first):
  * pkg_match   - the app's package ID appears in a Play link in the RBI list
  * name_score  - token_set_ratio(normalised title, normalised dla_name), best RBI row
  * dev_score   - token_set_ratio(normalised developer, normalised re_name / dla_owner)

Generic words ("loan", "cash", "instant", ...) are removed before fuzzy matching
so shared boilerplate does not inflate scores (e.g. "Cash" vs "Fast Cash").

offers_credit() is a keyword rule used for labelling only. It is never a model
feature (see the leakage rule in PLAN.md section 5).
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process

LEGAL = r"\b(private|pvt|limited|ltd|llp|inc|co|company|corporation|corp|india|technologies|technology|tech|services|solutions|fintech|financial|finance|finserv|capital|ventures)\b"
NOISE_PHRASES = [
    r"instant personal loan(s)?( app)?", r"personal loan app", r"loan app", r"online loan",
    r"\bapp\b", r"\bandroid\b", r"\bios\b", r"\bwebsite\b", r"\bplay ?store\b", r"\bapp ?store\b",
    r"\(formerly[^)]*\)", r"\(old ?name[^)]*\)",
]
GENERIC = {
    "loan", "loans", "cash", "credit", "money", "paisa", "paise", "rupee", "rupees", "instant",
    "personal", "quick", "fast", "easy", "emi", "online", "mobile", "digital", "lending", "lend",
    "the", "and", "for", "by", "of", "a", "in", "your", "my", "pay", "bank", "banking", "upi",
    "salary", "advance", "get", "now", "new", "best", "smart", "super", "plus", "pro", "lite",
}


def normalise(name: str, legal: bool = True) -> str:
    """Lowercase, strip accents, marketing phrases, punctuation and legal suffixes."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    # brand usually precedes " - " / ": " taglines; fall back to the full name if that is empty
    parts = re.split(r"\s[-–|:]\s|:\s", s)
    for cand in ([parts[0], s] if len(parts) > 1 else [s]):
        out = _clean(cand, legal)
        if out:
            return out
    return ""


def _clean(s: str, legal: bool) -> str:
    for p in NOISE_PHRASES:
        s = re.sub(p, " ", s)
    if legal:
        s = re.sub(LEGAL, " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def distinctive(norm: str) -> str:
    return " ".join(t for t in norm.split() if t not in GENERIC and len(t) > 1)


def name_similarity(a: str, b: str) -> float:
    """token_sort_ratio on distinctive tokens; plain ratio when either side is all generic.

    token_set_ratio (the plan's choice) scores 100 for any token subset, which
    matched "Gaadi Bazaar" to "My Money Bazaar" and "Union Bank" to "City Union
    Bank" in the dry run; token_sort_ratio penalises the unmatched tokens.
    """
    da, db = distinctive(a), distinctive(b)
    if not da or not db:
        return fuzz.ratio(a, b)
    return fuzz.token_sort_ratio(da, db)


def best_match(query: str, choices: list[str]) -> tuple[float, int | None]:
    if not query or not choices:
        return 0.0, None
    hit = process.extractOne(query, choices, scorer=lambda q, c, **_: name_similarity(q, c))
    return (float(hit[1]), hit[2]) if hit else (0.0, None)


def match_apps(apps: pd.DataFrame, rbi: pd.DataFrame) -> pd.DataFrame:
    rbi = rbi.copy()
    rbi["n_dla"] = rbi["dla_name"].map(lambda x: normalise(x, legal=False))
    rbi["n_re"] = rbi["re_name"].map(normalise)
    rbi["n_owner"] = rbi.get("dla_owner", pd.Series("", index=rbi.index)).map(normalise)
    pkgs = set(rbi["package_id"].dropna()) if "package_id" in rbi else set()

    dla_choices = rbi["n_dla"].tolist()
    ent_choices = pd.concat([rbi["n_re"], rbi["n_owner"]]).tolist()
    ent_rows = list(rbi.index) * 2

    out = []
    for r in apps.itertuples():
        n_title = normalise(r.title, legal=False)
        n_dev = normalise(r.developer)
        ns, ni = best_match(n_title, dla_choices)
        ds, di = best_match(n_dev, ent_choices)
        out.append({
            "app_id": r.app_id,
            "pkg_match": r.app_id in pkgs,
            "name_score": ns,
            "best_dla_name": rbi.at[rbi.index[ni], "dla_name"] if ni is not None else None,
            "best_dla_re": rbi.at[rbi.index[ni], "re_name"] if ni is not None else None,
            "dev_score": ds,
            "best_entity": (rbi.at[ent_rows[di], "re_name"] if di < len(rbi)
                            else rbi.at[ent_rows[di], "dla_owner"]) if di is not None else None,
        })
    return pd.DataFrame(out)


# --- does the app itself offer credit to users? -----------------------------
LENDING_PHRASES = [
    "loan amount", "apr", "annual percentage rate", "interest rate", "rate of interest", "tenure",
    "repayment", "disburs", "processing fee", "borrow", "nbfc", "credit limit", "credit line",
    "personal loan", "instant loan", "cash loan", "get a loan", "apply for loan", "apply for a loan",
    "loan up to", "loans up to", "loan of", "loan against", "gold loan", "business loan",
    "home loan", "pay later", "emi", "lending partner", "kfs", "key fact statement",
]
NON_LENDER_TITLE = [
    "calculator", "ledger", "khata", "bahi", "stock", "trading", "crypto", "bitcoin", "insurance",
    "tax", "gst", "invoice", "billing", "accounting", "news", "quiz", "guide", "tips", "advice",
    "info", "balance check", "mutual fund", "sip", "demat", "horoscope", "expense manager",
]


STAFF_TITLE = re.compile(
    r"\b(sales|partner|agent|distributor|dsa|employee|staff|collection|field|merchant|bridge|connect)\b", re.I)

STEMS = {"disburs", "borrow", "repayment"}  # match as prefixes (disbursal, borrowing, ...)


def _phrase_count(text: str) -> int:
    t = text.lower()
    return sum(1 for p in LENDING_PHRASES
               if re.search(rf"\b{re.escape(p)}" + ("" if p in STEMS else r"\b"), t))


def offers_credit(title: str, summary: str, description: str) -> tuple[str, str]:
    """Return ('yes'|'no'|'unsure', reason)."""
    head = f"{title or ''} {summary or ''}".lower()
    n = _phrase_count(f"{head} {description or ''}")
    cue = next((c for c in NON_LENDER_TITLE if re.search(rf"\b{re.escape(c)}\b", head)), None)
    if n == 0:
        return "no", "no lending phrases"
    if cue and n < 5:
        return "no", f"non-lender cue '{cue}', {n} lending phrases"
    if cue:
        return "unsure", f"non-lender cue '{cue}' but {n} lending phrases"
    if n >= 3:
        return "yes", f"{n} lending phrases"
    return "unsure", f"only {n} lending phrases"

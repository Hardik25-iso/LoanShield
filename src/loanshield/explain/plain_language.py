"""Turn model features into plain-language statements (PLAN.md section 7).

describe(feature, value) states what was observed about the app, e.g.
"Requests access to your contacts". Whether that pushed the score toward
"unverified" or away from it comes from the SHAP sign, not from this text.
Shared by the SHAP report and the Streamlit demo.
"""
from __future__ import annotations

import math

PERMISSION_TEXT = {
    "READ_CONTACTS": "your contacts",
    "WRITE_CONTACTS": "change your contacts",
    "READ_SMS": "your text messages (SMS)",
    "RECEIVE_SMS": "incoming text messages",
    "SEND_SMS": "send text messages",
    "READ_CALL_LOG": "your call history",
    "CALL_PHONE": "make phone calls directly",
    "CAMERA": "your camera",
    "RECORD_AUDIO": "your microphone",
    "ACCESS_FINE_LOCATION": "your precise location",
    "ACCESS_COARSE_LOCATION": "your approximate location",
    "READ_STORAGE": "your photos and files",
    "GET_ACCOUNTS": "the accounts on your device",
    "READ_PHONE_STATE": "your phone's identity and status",
    "READ_CALENDAR": "your calendar",
    "SYSTEM_ALERT_WINDOW": "draw over other apps",
}

INSTALL_BANDS = ["0", "1+", "5+", "10+", "50+", "100+", "500+", "1K+", "5K+", "10K+", "50K+",
                 "100K+", "500K+", "1M+", "5M+", "10M+", "50M+", "100M+", "500M+", "1B+", "5B+"]


def _pct(v) -> str:
    return f"{100 * v:.0f}%"


def _yes(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and v >= 0.5


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def describe(feature: str, value) -> str:
    """One sentence about the observed value of `feature`."""
    if feature.startswith("missingindicator_"):
        base = feature.removeprefix("missingindicator_")
        return describe(base, float("nan")) if _yes(value) else f"{LABELS.get(base, base).capitalize()} is available"
    if _isnan(value):
        if feature.startswith("rev_"):
            return "No user reviews yet"  # all review aggregates are NaN together
        if feature in {"score", "share_1star", "share_5star", "polarisation"}:
            return "Too few ratings for Play to show a score"
        return f"No data for {LABELS.get(feature, feature)}"

    if feature.startswith("perm_"):
        what = PERMISSION_TEXT.get(feature.removeprefix("perm_"), feature.removeprefix("perm_"))
        verb = "Requests permission to" if what.split()[0] in {"change", "send", "make", "draw"} else "Requests access to"
        return f"{verb} {what}" if _yes(value) else f"Does not request {what.replace('change ', '').replace('send ', '').replace('make ', '')}"

    fn = TEMPLATES.get(feature)
    if fn:
        return fn(value)
    return f"{LABELS.get(feature, feature)} = {value:.3g}"


LABELS = {
    "score": "Play Store rating", "share_1star": "share of 1-star ratings",
    "share_5star": "share of 5-star ratings", "polarisation": "rating polarisation",
    "app_age_days": "app age", "days_since_update": "last update date",
    "rev_mean_score": "review ratings", "rev_share_1star": "1-star reviews",
    "rev_share_harass": "harassment reports in reviews", "rev_share_fee": "hidden-fee complaints",
    "rev_share_fraud": "fraud complaints", "rev_mean_len": "review length", "contains_ads": "ads",
}

TEMPLATES = {
    "log_min_installs": lambda v: f"About {10 ** v - 1:,.0f}+ installs",
    "install_band": lambda v: f"Install band {INSTALL_BANDS[int(v)] if 0 <= int(v) < len(INSTALL_BANDS) else int(v)}",
    "score": lambda v: f"Rated {v:.1f} stars on Play",
    "has_score": lambda v: "Has a Play Store rating" if _yes(v) else "Too few ratings for a Play Store score",
    "log_ratings": lambda v: f"About {10 ** v - 1:,.0f} ratings",
    "share_1star": lambda v: f"{_pct(v)} of ratings are 1-star",
    "share_5star": lambda v: f"{_pct(v)} of ratings are 5-star",
    "polarisation": lambda v: f"{_pct(v)} of ratings are either 1-star or 5-star",
    "app_age_days": lambda v: f"Released {v / 365:.1f} years ago" if v >= 365 else f"Released {v:.0f} days ago",
    "days_since_update": lambda v: f"Last updated {v:.0f} days ago",
    "update_missing": lambda v: "Play does not show when the app was last updated" if _yes(v) else "Update date is shown",
    "contains_ads": lambda v: "Shows ads" if _yes(v) else "No ads",
    "has_privacy_policy": lambda v: "Has a privacy policy" if _yes(v) else "No privacy policy linked",
    "pp_domain_matches_site": lambda v: ("Privacy policy is on the developer's own website" if _yes(v)
                                         else "Privacy policy is not on the developer's own website"),
    "dev_app_count": lambda v: f"Developer has {v:.0f} lending app(s) in our dataset",
    "dev_email_free": lambda v: ("Developer uses a free email address (e.g. Gmail)" if _yes(v)
                                 else "Developer uses a company email address"),
    "dev_email_missing": lambda v: "No developer email" if _yes(v) else "Developer email is listed",
    "dev_email_matches_site": lambda v: ("Developer email matches its website" if _yes(v)
                                         else "Developer email does not match its website"),
    "dev_has_website": lambda v: "Developer has a website" if _yes(v) else "Developer has no website",
    "dev_near_dup_titles": lambda v: (f"Developer has {v:.0f} other app(s) with near-identical names" if v >= 1
                                      else "No near-duplicate apps from this developer"),
    "n_dangerous_perms": lambda v: f"Requests {v:.0f} sensitive permissions",
    "n_permission_labels": lambda v: f"Requests {v:.0f} permissions in total",
    "perms_missing": lambda v: "Permission list unavailable" if _yes(v) else "Permission list available",
    "mentions_partner": lambda v: ("Names an NBFC or bank lending partner" if _yes(v)
                                   else "Does not name an NBFC or bank lending partner"),
    "states_apr": lambda v: "States the interest rate / APR" if _yes(v) else "Does not state the interest rate / APR",
    "states_tenure": lambda v: "States the loan tenure" if _yes(v) else "Does not state the loan tenure",
    "mentions_kfs_grievance": lambda v: ("Mentions a Key Fact Statement or grievance officer" if _yes(v)
                                         else "No Key Fact Statement or grievance officer mentioned"),
    "urgency_terms": lambda v: (f"Uses {v:.0f} urgency phrase(s) (e.g. 'instant', 'no CIBIL')" if v >= 1
                                else "No urgency phrases"),
    "log_desc_len": lambda v: f"Description is about {10 ** v - 1:,.0f} characters",
    "has_reviews": lambda v: "Has user reviews" if _yes(v) else "No user reviews yet",
    "log_rev_count": lambda v: f"About {10 ** v - 1:,.0f} recent reviews analysed",
    "rev_mean_score": lambda v: f"Recent reviews average {v:.1f} stars",
    "rev_share_1star": lambda v: f"{_pct(v)} of recent reviews are 1-star",
    "rev_share_harass": lambda v: f"{_pct(v)} of recent reviews mention harassment, threats or contacting family",
    "rev_share_fee": lambda v: f"{_pct(v)} of recent reviews complain about hidden fees or deductions",
    "rev_share_fraud": lambda v: f"{_pct(v)} of recent reviews call it a fraud or scam",
    "rev_mean_len": lambda v: f"Reviews average {v:.0f} characters",
}


def top_signals(shap_row: dict, values: dict, k: int = 3, toward_unverified: bool = True) -> list[dict]:
    """k features with the largest SHAP push in one direction, described in words."""
    sign = 1 if toward_unverified else -1
    ranked = sorted(((f, s) for f, s in shap_row.items() if sign * s > 0), key=lambda t: -abs(t[1]))
    out, seen = [], {}
    for f, s in ranked:  # features describing the same fact (e.g. "No user reviews yet") merge
        t = describe(f, values.get(f))
        if t in seen:
            seen[t]["shap"] += s
            continue
        seen[t] = {"feature": f, "shap": s, "value": values.get(f), "text": t}
        out.append(seen[t])
        if len(out) == k:
            break
    return out

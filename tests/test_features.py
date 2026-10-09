import pandas as pd

from loanshield.features.metadata import _domain, install_band, listing_text_features
from loanshield.features.permissions import perm_flags
from loanshield.features.text import LEX_RE, clean_review, clean_reviews


def test_perm_flags_maps_labels_and_dedups_groups():
    p = {"Contacts": ["read your contacts"],
         "Phone": ["read phone status and identity"],
         "Device ID & call information": ["read phone status and identity"],
         "SMS": ["read your text messages (SMS or MMS)"],
         "Other": ["full network access"]}
    f = perm_flags(p)
    assert f["perm_READ_CONTACTS"] == 1 and f["perm_READ_SMS"] == 1 and f["perm_CAMERA"] == 0
    assert f["n_dangerous_perms"] == 3 and f["perms_missing"] == 0


def test_perm_flags_missing():
    f = perm_flags({"_missing": True})
    assert f["perms_missing"] == 1 and pd.isna(f["perm_READ_SMS"])


def test_clean_review():
    t = clean_review("Total FROUD!! call 9876543210 https://x.co 😡 bohot bura, nhi doonga paise")
    assert "9876543210" not in t and "http" not in t and "😡" not in t
    assert "fraud" in t and "bahut" in t and "nahi" in t and "paisa" in t


def test_lexicons():
    assert LEX_RE["harass"].search(clean_review("They called my family and relatives, abusing"))
    assert not LEX_RE["harass"].search(clean_review("please contact customer care"))
    assert LEX_RE["fee"].search(clean_review("processing fee deducted, got less amount"))
    assert LEX_RE["fraud"].search(clean_review("dhokhebaz app, chor log"))


def test_clean_reviews_dedups_within_app():
    rv = pd.DataFrame({"app_id": ["a", "a", "b"], "lang": "en", "content": ["Fraud!", "fraud", "fraud"],
                       "score": [1, 1, 1], "at": pd.to_datetime(["2026-01-01"] * 3),
                       "thumbsUpCount": 0, "appVersion": None})
    rc = clean_reviews(rv)
    assert len(rc) == 2 and rc["lex_fraud"].sum() == 2


def test_domain_and_bands():
    assert _domain("https://www.kissht.com/privacy") == "kissht.com"
    assert _domain("https://abc.co.in/x") == "abc.co.in"
    assert install_band(0) == 0 and install_band(1_000) < install_band(5_000)


def test_listing_text():
    apps = pd.DataFrame({"app_id": ["x"], "title": ["QuickCash"], "summary": ["Instant loan, no CIBIL"],
                         "description": ["Loans by our NBFC partner. APR 24% p.a. Tenure 91 to 365 days. "
                                         "Grievance Officer: Mr X"]})
    f = listing_text_features(apps).iloc[0]
    assert f.mentions_partner == 1 and f.states_apr == 1 and f.states_tenure == 1
    assert f.mentions_kfs_grievance == 1 and f.urgency_terms >= 2

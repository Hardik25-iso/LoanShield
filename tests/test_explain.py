import json

from loanshield.common import path
from loanshield.explain.plain_language import TEMPLATES, describe, top_signals


def test_describe_permissions_and_flags():
    assert describe("perm_READ_CONTACTS", 1) == "Requests access to your contacts"
    assert describe("perm_SEND_SMS", 1) == "Requests permission to send text messages"
    assert describe("perm_READ_SMS", 0).startswith("Does not request")
    assert describe("mentions_partner", 0) == "Does not name an NBFC or bank lending partner"
    assert describe("rev_share_harass", 0.12).startswith("12% of recent reviews")
    assert describe("missingindicator_score", 1) == "Too few ratings for Play to show a score"
    assert describe("rev_share_fee", float("nan")) == "No user reviews yet"
    assert describe("app_age_days", float("nan")) == "No data for app age"


def test_every_feature_has_a_sentence():
    f = path("processed", "feature_groups.json")
    if not f.exists():
        return
    for feat in json.loads(f.read_text(encoding="utf-8")):
        if not feat.startswith("perm_"):
            assert feat in TEMPLATES, f"no plain-language template for {feat}"


def test_top_signals_direction():
    shap_row = {"perm_READ_CONTACTS": 0.4, "mentions_partner": 0.2, "score": -0.5, "urgency_terms": 0.05}
    vals = {"perm_READ_CONTACTS": 1, "mentions_partner": 0, "score": 4.5, "urgency_terms": 2}
    up = top_signals(shap_row, vals, k=2)
    assert [s["feature"] for s in up] == ["perm_READ_CONTACTS", "mentions_partner"]
    down = top_signals(shap_row, vals, k=3, toward_unverified=False)
    assert [s["feature"] for s in down] == ["score"]


def test_top_signals_merges_duplicate_sentences():
    nan = float("nan")
    shap_row = {"rev_share_fee": -0.1, "rev_share_harass": -0.08, "score": -0.05, "dev_has_website": -0.01}
    vals = {"rev_share_fee": nan, "rev_share_harass": nan, "score": 4.0, "dev_has_website": 1}
    texts = [s["text"] for s in top_signals(shap_row, vals, 3, toward_unverified=False)]
    assert texts == ["No user reviews yet", "Rated 4.0 stars on Play", "Developer has a website"]

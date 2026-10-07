import pandas as pd

from loanshield.collect.play_details import filter_reason, lending_re
from loanshield.collect.rbi_directory import find_header, map_columns, norm_re_type


def test_find_header_skips_title_rows():
    raw = pd.DataFrame([
        ["DLAs deployed by Regulated Entities", None, None],
        [None, None, None],
        ["Name of RE", "Type of RE", "DLA Name"],
        ["ABC Finance Ltd", "NBFC", "QuickRupee"],
    ])
    df = map_columns(find_header(raw))
    assert list(df[["dla_name", "re_name", "re_type"]].iloc[0]) == ["QuickRupee", "ABC Finance Ltd", "NBFC"]


def test_norm_re_type():
    assert norm_re_type("Scheduled Commercial Bank") == "bank"
    assert norm_re_type("NBFC-ICC") == "NBFC"
    assert norm_re_type("Housing Finance Company") == "HFC"
    assert norm_re_type("Cooperative Society") == "other"


def test_filter_reason():
    pat = lending_re()
    assert filter_reason({"_not_found": True}, pat) == "not_found_on_play"
    assert filter_reason({"genre": "Tools", "title": "loan"}, pat).startswith("genre:")
    assert filter_reason({"genre": "Finance", "title": "Stock tips"}, pat) == "no_lending_terms"
    assert filter_reason({"genre": "Finance", "title": "X", "description": "Instant EMI"}, pat) == "keep"

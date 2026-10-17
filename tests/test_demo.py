import numpy as np
import pandas as pd
import pytest

from loanshield.common import path
from loanshield.demo import build_features, load_cached, parse_app_id, risk_band


def test_parse_app_id():
    assert parse_app_id("https://play.google.com/store/apps/details?id=com.pocketly&hl=en_IN") == "com.pocketly"
    assert parse_app_id("https://play.google.com/store/apps/details?hl=en&id=in.foo.bar_2") == "in.foo.bar_2"
    assert parse_app_id("  com.kreditbee.android ") == "com.kreditbee.android"
    assert parse_app_id("kreditbee") is None
    assert parse_app_id("https://example.com/not-play") is None


def test_risk_band_order():
    thr = 0.3
    assert [risk_band(p, thr) for p in (0.05, 0.2, 0.35, 0.8)] == ["low", "moderate", "elevated", "high"]


@pytest.mark.skipif(not path("processed", "features.parquet").exists(), reason="features not built")
def test_demo_features_match_training_features():
    """Training/serving skew check: cached apps must get exactly their training features."""
    train = pd.read_parquet(path("processed", "features.parquet")).set_index("app_id")
    context = pd.read_csv(path("interim", "apps.csv"))
    ids = train.sample(25, random_state=42).index
    for aid in ids:
        X, _ = build_features(load_cached(aid), context, list(train.columns))
        a, b = X.iloc[0].to_numpy(float), train.loc[aid].to_numpy(float)
        bad = ~np.isclose(a, b, equal_nan=True)
        assert not bad.any(), f"{aid}: {dict(zip(train.columns[bad], zip(a[bad], b[bad])))}"


@pytest.mark.skipif(not (path("processed", "features.parquet").exists()
                         and (path("raw").parent / "demo" / "offline_bundle.json.gz").exists()),
                    reason="features or offline bundle not built")
def test_offline_bundle_gives_training_features(monkeypatch):
    """Fresh clone / cloud: raw caches absent, the committed bundle must reproduce the features."""
    import loanshield.demo as demo
    real_path = demo.path
    monkeypatch.setattr(demo, "path", lambda key, *p: (real_path(key, *p).with_name("__absent__")
                                                       if key == "raw" else real_path(key, *p)))
    train = pd.read_parquet(real_path("processed", "features.parquet")).set_index("app_id")
    context = pd.read_csv(real_path("interim", "apps.csv"))
    for aid in train.sample(10, random_state=7).index:
        a = demo.load_cached(aid)
        X, _ = build_features(a, context, list(train.columns))
        assert np.allclose(X.iloc[0].to_numpy(float), train.loc[aid].to_numpy(float), equal_nan=True), aid

import pandas as pd

import loanshield.label.build_labels as bl


def _setup(tmp_path, monkeypatch, a, b=None, resolved=None):
    monkeypatch.setattr(bl, "path", lambda _key, name: tmp_path / name)
    pd.DataFrame({"app_id": list(a), "label": list(a.values())}).to_csv(tmp_path / "label_review_A.csv", index=False)
    if b is not None:
        pd.DataFrame({"app_id": list(b), "label": list(b.values())}).to_csv(tmp_path / "label_review_B.csv", index=False)
    if resolved is not None:
        pd.DataFrame({"app_id": list(resolved), "label": list(resolved.values())}).to_csv(
            tmp_path / "label_review_resolved.csv", index=False)


QUEUE = pd.DataFrame({"app_id": ["a", "b", "c", "d"]})
A = {"a": "verified", "b": "unverified", "c": "not_lender", "d": "unverified"}


def test_single_annotator_reports_no_kappa(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, A)
    final, stats = bl.load_manual(QUEUE)
    assert final.to_dict() == A and stats["status"] == "single" and stats["kappa"] is None


def test_partial_b_kappa_on_overlap_and_a_elsewhere(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, A, b={"a": "verified", "b": "not_lender", "c": ""})  # c left blank by B
    final, stats = bl.load_manual(QUEUE)
    assert stats["n_double"] == 2 and stats["agreement"] == 0.5 and stats["disagreements"] == 1
    assert stats["unresolved"] == 1 and final["b"] == "unverified"  # unresolved keeps A's label
    assert final.to_dict() == A


def test_resolution_overrides_disagreement(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, A, b={"a": "verified", "b": "not_lender"}, resolved={"b": "not_lender"})
    final, stats = bl.load_manual(QUEUE)
    assert final["b"] == "not_lender" and stats["unresolved"] == 0

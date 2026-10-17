"""MuRIL on CPU: frozen embeddings + logistic regression (PLAN.md 6.3 fallback).

No GPU was available for fine-tuning, so, as the plan specifies, MuRIL is used
frozen: each review is embedded once (mean-pooled last hidden state, max_len 128),
then a class-weighted logistic regression is trained per developer-grouped fold
on the training apps' reviews, and an app's score is the mean P(unverified) over
its reviews. The folds are identical to the main experiment's.

Embeddings are cached by review text in data/interim/muril_embeddings.npz, so
re-labelling does not re-embed. To bound CPU time, at most the newest
MAX_REVIEWS_PER_APP reviews per app are used (MuRIL only).

Output: results/oof/<model>[_masked]_grouped.csv (app_id, score), which --exp main picks up.
Encoders: muril (google/muril-base-cased) and minilm (paraphrase-multilingual-MiniLM-L12-v2).
Embeddings are standardised before the LR (unscaled activations hurt the first MuRIL run).

Run: python -m loanshield.models.muril_frozen --model {muril,minilm} [--mask]
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from loanshield.common import cfg, get_logger, path
from loanshield.models.splits import grouped_folds, load_dataset
from loanshield.models.text_models import export_folds

log = get_logger("frozen_text")
# encoder key -> Hugging Face model names (first that loads wins)
ENCODERS = {
    "muril": ["google/muril-base-cased", "ai4bharat/indic-bert"],
    # sentence-level multilingual encoder (50+ languages incl. Hindi); built for mean pooling
    "minilm": ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
}
MAX_LEN, BATCH, MAX_REVIEWS_PER_APP = 128, 64, 100


def load_encoder(key: str = "muril"):
    from transformers import AutoModel, AutoTokenizer
    for name in ENCODERS[key]:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            enc = AutoModel.from_pretrained(name).eval()
            log.info("loaded %s", name)
            return name, tok, enc
        except Exception as e:  # noqa: BLE001 - try the fallback model
            log.warning("could not load %s: %s", name, e)
    raise RuntimeError(f"no {key} encoder could be loaded")


def embed(texts: list[str], key: str = "muril") -> tuple[str, dict[str, np.ndarray]]:
    """Embeddings for `texts`, reusing the on-disk cache (one cache file per encoder)."""
    import torch
    f = path("interim", f"{key}_embeddings.npz")
    cache: dict[str, np.ndarray] = {}
    model_name = None
    if f.exists():
        z = np.load(f, allow_pickle=True)
        model_name = str(z["model"])
        cache = dict(zip(z["texts"].tolist(), z["vectors"]))
    todo = sorted(set(texts) - set(cache))
    log.info("embeddings: %d cached, %d to compute", len(set(texts)) - len(todo), len(todo))
    if todo:
        name, tok, enc = load_encoder(key)
        if model_name and model_name != name:
            raise RuntimeError(f"cache built with {model_name}, now loading {name}; delete {f}")
        model_name = name
        torch.set_num_threads(max(1, torch.get_num_threads()))
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(todo), BATCH):
                batch = todo[i:i + BATCH]
                b = tok(batch, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
                h = enc(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                vec = ((h * m).sum(1) / m.sum(1)).numpy().astype(np.float32)
                cache.update(zip(batch, vec))
                if (i // BATCH) % 50 == 0:
                    done = i + len(batch)
                    rate = done / (time.time() - t0)
                    log.info("embedded %d/%d (%.0f/s, ~%.0f min left)", done, len(todo), rate,
                             (len(todo) - done) / rate / 60)
                if (i // BATCH) % 200 == 199:  # checkpoint long runs
                    _save(f, model_name, cache)
        _save(f, model_name, cache)
    return model_name, cache


def _save(f, model_name, cache) -> None:
    keys = list(cache)
    np.savez(f, model=model_name, texts=np.array(keys, dtype=object), vectors=np.stack([cache[k] for k in keys]))


def oof_scores(key: str = "muril", mask: bool = False) -> pd.Series:
    """Out-of-fold app scores from frozen `key` embeddings + scaled, class-weighted LR."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from loanshield.models.text_models import mask_reviews
    ds = load_dataset("features")
    folds = grouped_folds(ds)
    export_folds(ds, folds, "grouped")
    rc = pd.read_parquet(path("processed", "reviews_clean.parquet"), columns=["app_id", "text", "at"])
    rc = rc[rc["app_id"].isin(ds.X.index)].sort_values("at", ascending=False)
    rc = rc.groupby("app_id").head(MAX_REVIEWS_PER_APP).reset_index(drop=True)
    if mask:
        rc = mask_reviews(rc)
    log.info("%s%s: %d reviews from %d apps", key, " (brand-masked)" if mask else "", len(rc), rc["app_id"].nunique())

    model_name, cache = embed(rc["text"].tolist(), key)
    E = np.stack([cache[t] for t in rc["text"]])
    y = rc["app_id"].map(ds.y).to_numpy()

    oof = pd.Series(np.nan, index=ds.X.index)
    for k, (tr, te) in enumerate(folds):
        tr_m = rc["app_id"].isin(ds.X.index[tr]).to_numpy()
        te_m = rc["app_id"].isin(ds.X.index[te]).to_numpy()
        # embeddings are unscaled mean-pooled activations: standardise before the LR
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=5000, random_state=cfg()["seed"]))
        clf.fit(E[tr_m], y[tr_m])
        p = pd.Series(clf.predict_proba(E[te_m])[:, 1], index=rc.index[te_m])
        s = p.groupby(rc.loc[te_m, "app_id"]).mean()
        oof.loc[s.index] = s.values
        log.info("fold %d: %d test apps scored", k, len(s))
    log.info("%s (frozen) OOF scores for %d apps", model_name, int(oof.notna().sum()))
    return oof


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(ENCODERS), default="muril")
    ap.add_argument("--mask", action="store_true", help="mask each app's own brand names in its reviews")
    a = ap.parse_args()
    oof = oof_scores(a.model, a.mask)
    out = path("results", "oof", f"{a.model}{'_masked' if a.mask else ''}_grouped.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    oof.dropna().rename("score").rename_axis("app_id").reset_index().to_csv(out, index=False)
    log.info("-> %s", out)


if __name__ == "__main__":
    main()

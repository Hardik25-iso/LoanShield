"""MuRIL on CPU: frozen embeddings + logistic regression (PLAN.md 6.3 fallback).

No GPU was available for fine-tuning, so, as the plan specifies, MuRIL is used
frozen: each review is embedded once (mean-pooled last hidden state, max_len 128),
then a class-weighted logistic regression is trained per developer-grouped fold
on the training apps' reviews, and an app's score is the mean P(unverified) over
its reviews. The folds are identical to the main experiment's.

Embeddings are cached by review text in data/interim/muril_embeddings.npz, so
re-labelling does not re-embed. To bound CPU time, at most the newest
MAX_REVIEWS_PER_APP reviews per app are used (MuRIL only).

Output: results/oof/muril_grouped.csv (app_id, score), which --exp main picks up.

Run: python -m loanshield.models.muril_frozen
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from loanshield.common import cfg, get_logger, path
from loanshield.models.splits import grouped_folds, load_dataset
from loanshield.models.text_models import export_folds

log = get_logger("muril")
MODELS = ["google/muril-base-cased", "ai4bharat/indic-bert"]
MAX_LEN, BATCH, MAX_REVIEWS_PER_APP = 128, 64, 100
CACHE = "muril_embeddings.npz"


def load_encoder():
    from transformers import AutoModel, AutoTokenizer
    for name in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            enc = AutoModel.from_pretrained(name).eval()
            log.info("loaded %s", name)
            return name, tok, enc
        except Exception as e:  # noqa: BLE001 - try the fallback model
            log.warning("could not load %s: %s", name, e)
    raise RuntimeError("no MuRIL / IndicBERT model could be loaded")


def embed(texts: list[str]) -> tuple[str, dict[str, np.ndarray]]:
    """Embeddings for `texts`, reusing the on-disk cache."""
    import torch
    f = path("interim", CACHE)
    cache: dict[str, np.ndarray] = {}
    model_name = None
    if f.exists():
        z = np.load(f, allow_pickle=True)
        model_name = str(z["model"])
        cache = dict(zip(z["texts"].tolist(), z["vectors"]))
    todo = sorted(set(texts) - set(cache))
    log.info("embeddings: %d cached, %d to compute", len(set(texts)) - len(todo), len(todo))
    if todo:
        name, tok, enc = load_encoder()
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


def main() -> None:
    ds = load_dataset("features")
    folds = grouped_folds(ds)
    export_folds(ds, folds, "grouped")
    rc = pd.read_parquet(path("processed", "reviews_clean.parquet"), columns=["app_id", "text", "at"])
    rc = rc[rc["app_id"].isin(ds.X.index)].sort_values("at", ascending=False)
    rc = rc.groupby("app_id").head(MAX_REVIEWS_PER_APP).reset_index(drop=True)
    log.info("%d reviews from %d apps", len(rc), rc["app_id"].nunique())

    model_name, cache = embed(rc["text"].tolist())
    E = np.stack([cache[t] for t in rc["text"]])
    y = rc["app_id"].map(ds.y).to_numpy()

    oof = pd.Series(np.nan, index=ds.X.index)
    for k, (tr, te) in enumerate(folds):
        tr_m = rc["app_id"].isin(ds.X.index[tr]).to_numpy()
        te_m = rc["app_id"].isin(ds.X.index[te]).to_numpy()
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=cfg()["seed"])
        clf.fit(E[tr_m], y[tr_m])
        p = pd.Series(clf.predict_proba(E[te_m])[:, 1], index=rc.index[te_m])
        s = p.groupby(rc.loc[te_m, "app_id"]).mean()
        oof.loc[s.index] = s.values
        log.info("fold %d: %d test apps scored", k, len(s))

    out = path("results", "oof", "muril_grouped.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    oof.dropna().rename("score").rename_axis("app_id").reset_index().to_csv(out, index=False)
    log.info("%s (frozen) OOF scores for %d apps -> %s", model_name, int(oof.notna().sum()), out)


if __name__ == "__main__":
    main()

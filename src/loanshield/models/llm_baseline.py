"""Zero-shot LLM baseline: Claude reads each app's Play listing and estimates the
probability that the app is NOT in RBI's Digital Lending Apps directory.

No training, so every score is effectively out-of-sample and is evaluated on the same
apps and folds as the other models (it is picked up by `run_experiments --exp main`
as the external model "llm").

Caveat for the report: an LLM may recognise well-known brands from its own training
data, so its score mixes listing evidence with prior brand knowledge. The prompt asks
it to rely on the listing, but that cannot be enforced.

Each app's response is cached in data/interim/llm_scores/<app_id>.json, so a re-run
only calls the API for apps not yet scored.

Credentials: ANTHROPIC_API_KEY (or an `ant auth login` profile).
Cost: about 414 apps x ~2.5k input tokens at claude-opus-5-5 rates (~$4-5 total);
run `--dry-run` first to see the prompt and an estimate without calling the API.

Run: python -m loanshield.models.llm_baseline [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from loanshield.common import get_logger, path, read_json, safe_name, write_json
from loanshield.explain.plain_language import PERMISSION_TEXT

log = get_logger("llm_baseline")
MODEL = "claude-opus-5-5"
N_REVIEWS = 8

SYSTEM = """You assess Indian lending apps listed on Google Play for a research study.

Since July 2025 the Reserve Bank of India (RBI) publishes a directory of Digital Lending Apps (DLAs) deployed by regulated entities (banks, NBFCs, housing finance companies) and their lending service providers. You will see one app's public Play Store listing, its requested permissions and a few recent reviews.

Estimate the probability that this specific app is NOT in the RBI DLA directory. Base the estimate on the evidence in the listing: whether a regulated lending partner is named, whether loan terms (APR, tenure, fees) and grievance details are disclosed, the developer's identity, the permissions requested, and what users report. Do not rely on whether you recognise the brand.

Return the probability and up to three short pieces of evidence."""

SCHEMA = {
    "type": "object",
    "properties": {
        "p_unlisted": {"type": "number", "description": "probability between 0 and 1"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["p_unlisted", "evidence"],
    "additionalProperties": False,
}


def app_prompt(app: pd.Series, perms: list[str], reviews: list[str]) -> str:
    rv = "\n".join(f"- {t[:300]}" for t in reviews) or "(no reviews)"
    return (f"Title: {app.title}\nDeveloper: {app.developer}\nDeveloper email: {app.developerEmail}\n"
            f"Developer website: {app.developerWebsite}\nInstalls: {app.installs}\nReleased: {app.released}\n"
            f"Summary: {app.summary}\n\nDescription:\n{app.description}\n\n"
            f"Requested access to: {', '.join(perms) or '(none of the sensitive permissions)'}\n\n"
            f"Recent reviews:\n{rv}")


def load_inputs() -> list[tuple[str, str]]:
    """(app_id, prompt) for every labelled app."""
    labels = pd.read_csv(path("processed", "labels.csv"))
    apps = pd.read_csv(path("interim", "apps.csv")).set_index("app_id")
    feats = pd.read_parquet(path("processed", "features.parquet")).set_index("app_id")
    rc = pd.read_parquet(path("processed", "reviews_clean.parquet"), columns=["app_id", "text", "at"])
    rc = rc.sort_values("at", ascending=False).groupby("app_id")["text"].apply(lambda s: list(s.head(N_REVIEWS)))
    out = []
    for aid in labels["app_id"]:
        f = feats.loc[aid]
        perms = [w for p, w in PERMISSION_TEXT.items() if f.get(f"perm_{p}", 0) == 1]
        out.append((aid, app_prompt(apps.loc[aid], perms, rc.get(aid, []))))
    return out


def score_one(client, aid: str, prompt: str) -> dict:
    f = path("interim", "llm_scores", f"{safe_name(aid)}.json")
    if f.exists():
        return read_json(f)
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # re-run on a fallback model if a safety classifier declines
    )
    if response.stop_reason == "refusal":
        res = {"app_id": aid, "p_unlisted": None, "evidence": [], "stop_reason": "refusal", "model": response.model}
    else:
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        res = {"app_id": aid, "p_unlisted": min(max(float(data["p_unlisted"]), 0.0), 1.0),
               "evidence": data["evidence"], "stop_reason": response.stop_reason, "model": response.model,
               "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    write_json(f, res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print one prompt and a cost estimate; no API calls")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    inputs = load_inputs()[: a.limit]
    if a.dry_run:
        chars = sum(len(SYSTEM) + len(p) for _, p in inputs)
        est_tokens = chars / 3.5
        print(f"{len(inputs)} apps, ~{est_tokens / 1e6:.2f}M input tokens -> ~${est_tokens / 1e6 * 4:.2f} input "
              f"at {MODEL} ($4/M), plus a few hundred output tokens per app ($20/M).\n")
        print("--- example prompt ---\n" + inputs[0][1][:1500])
        return

    import anthropic
    client = anthropic.Anthropic()
    rows = []
    for i, (aid, prompt) in enumerate(inputs, 1):
        try:
            rows.append(score_one(client, aid, prompt))
        except anthropic.RateLimitError:
            log.warning("rate limited at %s; re-run later to continue (scores are cached)", aid)
            break
        except anthropic.APIStatusError as e:
            log.warning("%s failed: %s", aid, e.message)
        if i % 25 == 0:
            log.info("%d/%d scored", i, len(inputs))
    df = pd.DataFrame(rows)
    out = path("results", "oof", "llm_grouped.csv")
    df.dropna(subset=["p_unlisted"]).rename(columns={"p_unlisted": "score"})[["app_id", "score"]].to_csv(out, index=False)
    log.info("%d apps scored (%d refusals) -> %s", df["p_unlisted"].notna().sum(),
             int((df.get("stop_reason") == "refusal").sum()), out)


if __name__ == "__main__":
    main()

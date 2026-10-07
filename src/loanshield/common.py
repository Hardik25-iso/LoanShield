"""Shared helpers: config, paths, polite scraping, JSON cache, logging."""
from __future__ import annotations

import json
import logging
import random
import time
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache
def cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str, *parts: str) -> Path:
    return ROOT / cfg()["paths"][key] / Path(*parts) if parts else ROOT / cfg()["paths"][key]


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def polite_sleep() -> None:
    s = cfg()["scrape"]
    time.sleep(s["sleep_min"] + random.uniform(0, s["sleep_jitter"]))


class NotFound(Exception):
    """The remote resource does not exist; do not retry."""


def with_retries(fn: Callable[[], Any], what: str, log: logging.Logger) -> Any:
    """Call fn with exponential backoff. NotFound is re-raised immediately."""
    s = cfg()["scrape"]
    for attempt in range(s["max_retries"]):
        try:
            return fn()
        except NotFound:
            raise
        except Exception as e:  # noqa: BLE001 - scraper raises many types
            if type(e).__name__ == "NotFoundError":
                raise NotFound(str(e)) from e
            wait = s["backoff_base"] * 2**attempt + random.uniform(0, 1)
            log.warning("%s failed (%s: %s); retry %d in %.0fs",
                        what, type(e).__name__, e, attempt + 1, wait)
            time.sleep(wait)
    raise RuntimeError(f"{what}: gave up after {s['max_retries']} attempts")


def _json_default(o: Any) -> Any:
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    raise TypeError(type(o))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=_json_default), encoding="utf-8")
    tmp.replace(p)


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def safe_name(app_id: str) -> str:
    return app_id.replace("/", "_")

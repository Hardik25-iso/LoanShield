"""Permission features.

Play's permissions() returns human-readable labels grouped by category, not
manifest constants, so each label is mapped to the Android permission it
describes. The same label can appear under two groups (e.g. "read phone status
and identity" under Phone and Device ID); flags are set once per app.
"""
from __future__ import annotations

import pandas as pd

from loanshield.common import path, read_json, safe_name

# label (lowercase) -> feature name. Plan list first, then extra high-risk ones.
LABEL_TO_PERM = {
    "read your contacts": "READ_CONTACTS",
    "read your text messages (sms or mms)": "READ_SMS",
    "receive text messages (sms)": "RECEIVE_SMS",
    "read call log": "READ_CALL_LOG",
    "take pictures and videos": "CAMERA",
    "precise location (gps and network-based)": "ACCESS_FINE_LOCATION",
    "read the contents of your usb storage": "READ_STORAGE",
    "read image files from shared storage": "READ_STORAGE",
    "read video files from shared storage": "READ_STORAGE",
    "find accounts on the device": "GET_ACCOUNTS",
    "read phone status and identity": "READ_PHONE_STATE",
    # extras: abuse-relevant capabilities seen in the data
    "send sms messages": "SEND_SMS",
    "directly call phone numbers": "CALL_PHONE",
    "record audio": "RECORD_AUDIO",
    "draw over other apps": "SYSTEM_ALERT_WINDOW",
    "read calendar events plus confidential information": "READ_CALENDAR",
    "modify your contacts": "WRITE_CONTACTS",
    "approximate location (network-based)": "ACCESS_COARSE_LOCATION",
}
PERMS = sorted(set(LABEL_TO_PERM.values()))
# Android "dangerous" protection level among the mapped permissions
DANGEROUS = {"READ_CONTACTS", "READ_SMS", "RECEIVE_SMS", "READ_CALL_LOG", "CAMERA",
             "ACCESS_FINE_LOCATION", "ACCESS_COARSE_LOCATION", "READ_STORAGE", "GET_ACCOUNTS",
             "READ_PHONE_STATE", "SEND_SMS", "CALL_PHONE", "RECORD_AUDIO", "READ_CALENDAR",
             "WRITE_CONTACTS"}


def perm_flags(perm_json: dict) -> dict:
    """Feature dict for one app's permissions() payload."""
    if not perm_json or perm_json.get("_missing"):
        return {**{f"perm_{p}": float("nan") for p in PERMS},
                "n_dangerous_perms": float("nan"), "n_permission_labels": float("nan"),
                "perms_missing": 1}
    labels = {lab.strip().lower() for labs in perm_json.values() for lab in labs}
    have = {LABEL_TO_PERM[lab] for lab in labels if lab in LABEL_TO_PERM}
    out = {f"perm_{p}": int(p in have) for p in PERMS}
    out["n_dangerous_perms"] = len(have & DANGEROUS)
    out["n_permission_labels"] = len(labels)
    out["perms_missing"] = 0
    return out


def build(app_ids: pd.Series) -> pd.DataFrame:
    rows = []
    for aid in app_ids:
        f = path("raw", "play_permissions", f"{safe_name(aid)}.json")
        rows.append({"app_id": aid, **perm_flags(read_json(f) if f.exists() else {})})
    return pd.DataFrame(rows)

"""LoanShield demo: is this Indian lending app in RBI's Digital Lending Apps directory?

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from loanshield.demo import (RBI_DLA_URL, RBI_HOME_URL, SACHET_URL, Predictor, cached_apps,
                             fetch_live, load_cached, parse_app_id, risk_band)

st.set_page_config(page_title="LoanShield", page_icon="🛡️", layout="wide")

# status colours (fixed palette): always paired with an icon and a text label
STATUS = {
    "high": ("#d03b3b", "⛔", "High risk"),
    "elevated": ("#ec835a", "⚠️", "Elevated risk"),
    "moderate": ("#fab219", "◐", "Moderate risk"),
    "low": ("#0ca30c", "✅", "Low risk"),
}

st.markdown("""
<style>
.ls-card { border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }
.ls-muted { opacity: .7; font-size: .9rem; }
.ls-meter { position: relative; height: 14px; border-radius: 7px; margin: 10px 0 4px;
            background: linear-gradient(90deg, #0ca30c 0%, #fab219 45%, #ec835a 70%, #d03b3b 100%); }
.ls-pin { position: absolute; top: -6px; width: 4px; height: 26px; border-radius: 2px;
          background: currentColor; box-shadow: 0 0 0 2px rgba(255,255,255,.8); }
.ls-thr { position: absolute; top: -3px; width: 0; height: 20px; border-left: 2px dashed rgba(128,128,128,.9); }
.ls-scale { display: flex; justify-content: space-between; font-size: .75rem; opacity: .7; }
.ls-badge { display: inline-block; padding: 6px 12px; border-radius: 999px; font-weight: 600;
            border: 2px solid; margin: 6px 0; }
.ls-review { border-left: 3px solid rgba(128,128,128,.4); padding: 4px 10px; margin: 8px 0; }
.ls-tag { display: inline-block; font-size: .72rem; padding: 1px 8px; border-radius: 999px;
          border: 1px solid rgba(128,128,128,.5); margin-right: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models…")
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data(ttl=3600, show_spinner=False)
def live(app_id: str):
    return fetch_live(app_id)


@st.cache_data(show_spinner=False)
def offline_list() -> pd.DataFrame:
    return cached_apps()


# --- sidebar ------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ LoanShield")
    st.caption("Checks whether an Indian lending app looks like an RBI-listed digital lending app.")
    mode = st.radio("Data source", ["Live from Google Play", "Offline (collected dataset)"],
                    help="Use Offline if the network is down. It scores apps from the cached dataset.")
    app_id, run = None, False
    if mode.startswith("Live"):
        raw = st.text_input("Play Store URL or package ID",
                            placeholder="https://play.google.com/store/apps/details?id=…")
        run = st.button("Check app", type="primary", use_container_width=True)
        if run:
            app_id = parse_app_id(raw)
            if not app_id:
                st.error("That doesn't look like a Play Store link or package ID (e.g. com.example.app).")
                run = False
    else:
        apps = offline_list()
        label = {"verified": "✓ verified", "unverified": "✗ unverified"}
        opts = apps.apply(lambda r: f"{r.title} — {r.developer}"
                          + (f"  [{label.get(r.label, '')}]" if isinstance(r.get("label"), str) else ""), axis=1)
        pick = st.selectbox(f"Choose one of {len(apps)} apps", range(len(apps)), format_func=lambda i: opts[i],
                            index=None, placeholder="Search by name…")
        run = st.button("Check app", type="primary", use_container_width=True, disabled=pick is None)
        if run and pick is not None:
            app_id = apps.loc[pick, "app_id"]
    st.divider()
    st.markdown(f"[RBI DLA directory]({RBI_DLA_URL}) · [RBI website]({RBI_HOME_URL}) · "
                f"[Sachet (report a lender)]({SACHET_URL})")

try:
    predictor = get_predictor()
except FileNotFoundError as e:
    st.error(f"Models not found. {e}")
    st.stop()

if not run or not app_id:
    st.header("Is this loan app RBI-listed?")
    st.write("Paste a Google Play link in the sidebar (or pick an app offline). LoanShield reads the app's "
             "listing, permissions and recent reviews and estimates how likely it is to be **missing from "
             "RBI's Digital Lending Apps directory**, with the reasons behind the estimate.")
    if predictor.card.get("n_unverified", 100) < 100:
        st.warning(f"Preliminary model: trained on only {predictor.card.get('n_unverified')} unverified apps. "
                   "Scores are indicative only.", icon="🧪")
    st.stop()

# --- score ----------------------------------------------------------------------
try:
    with st.spinner(f"Fetching {app_id} from Google Play…" if mode.startswith("Live") else "Loading…"):
        data = live(app_id) if mode.startswith("Live") else load_cached(app_id)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # noqa: BLE001 - network / scraper failures
    st.error(f"Couldn't reach Google Play ({type(e).__name__}). Switch to **Offline** in the sidebar "
             "to use the collected dataset.")
    st.stop()

with st.spinner("Scoring…"):
    r = predictor.predict(data)

band = risk_band(r["probability"], r["threshold"])
colour, icon, band_label = STATUS[band]
verdict = ("Unverified – review before use" if r["verdict"] == "unverified" else "Likely verified")

# header
st.subheader(r["title"] or app_id)
meta = [f"by {html.escape(str(r['developer']))}", f"<code>{html.escape(app_id)}</code>"]
if r["installs"]:
    meta.append(f"{r['installs']} installs")
if r["score"]:
    meta.append(f"★ {r['score']:.1f}")
meta.append("live from Google Play" if r["source"] == "live" else "offline dataset")
st.markdown(f'<div class="ls-muted">{" · ".join(meta)}</div>', unsafe_allow_html=True)
if r["preliminary"]:
    st.warning(f"Preliminary model (trained on {r['card'].get('n_unverified')} unverified apps). "
               "Treat this as indicative.", icon="🧪")

left, right = st.columns([3, 2], gap="large")
with left:
    pct = r["probability"] * 100
    st.markdown(f"""
<div class="ls-card">
  <div class="ls-muted">Estimated chance this app is <b>not</b> in RBI's DLA directory</div>
  <div style="font-size:2.6rem;font-weight:700;line-height:1.1">{pct:.0f}%</div>
  <div class="ls-meter" role="img" aria-label="risk {pct:.0f} percent">
    <div class="ls-thr" style="left:{r['threshold'] * 100:.1f}%" title="decision threshold"></div>
    <div class="ls-pin" style="left:calc({pct:.1f}% - 2px)"></div>
  </div>
  <div class="ls-scale"><span>0%</span><span>dashed line = decision threshold ({r['threshold'] * 100:.0f}%)</span><span>100%</span></div>
  <div class="ls-badge" style="border-color:{colour}">{icon} {band_label} · {verdict}</div>
</div>""", unsafe_allow_html=True)

with right:
    if r["rbi_listed"]:
        e = r["rbi_entries"][0]
        st.success(f"**Found in the RBI DLA directory** (snapshot {r['rbi_snapshot']}): "
                   f"{e['dla_name']}, deployed by {e['re_name']} ({e['re_type']}).", icon="🏛️")
    else:
        st.info(f"**Package not found** in our RBI DLA snapshot ({r['rbi_snapshot']}). The list changes; "
                f"[check the live directory]({RBI_DLA_URL}).", icon="🔎")
    st.caption("The directory lookup is a direct check. The percentage is the model's estimate from the "
               "app's own signals. It is useful for apps not yet listed or listed under another name.")

# signals
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("#### ⚠️ What raised the risk estimate")
    if r["warnings"]:
        for s in r["warnings"]:
            st.markdown(f"- {s['text']}")
    else:
        st.write("Nothing about this app pushed the estimate up.")
with c2:
    st.markdown("#### ↓ What lowered the risk estimate")
    if r["reassuring"]:
        for s in r["reassuring"]:
            st.markdown(f"- {s['text']}")
    else:
        st.write("None.")

st.caption("These are the model's top 3 reasons in each direction (SHAP values), compared with the typical "
           "lending app in our dataset. They describe what the model learned from the data, not a judgement "
           "that a signal is good or bad: e.g. large RBI-listed lenders also attract harassment complaints.")

# permissions
st.markdown("#### 🔐 Permissions")
if r["permissions_missing"]:
    st.write("Google Play did not return this app's permission list.")
else:
    perms = pd.DataFrame(r["permissions"])
    perms["Requested"] = perms["requested"].map({True: "⚠️ Yes", False: "No"})
    perms = perms.sort_values("requested", ascending=False)
    st.dataframe(perms.rename(columns={"what": "Access to", "permission": "Android permission"})
                 [["Access to", "Requested", "Android permission"]], hide_index=True, use_container_width=True)
    st.caption("Many RBI-listed lenders also request SMS and location access for credit checks, "
               "so a permission alone is not proof of a bad app. Contacts and call-log access deserve the most scrutiny.")

# reviews
st.markdown("#### 💬 Recent reviews flagged for harassment, hidden fees or fraud")
if not r["n_reviews"]:
    st.write("This app has no reviews to analyse yet.")
else:
    st.caption(f"{r['n_flagged']} of {r['n_reviews']} recent unique reviews match a warning term."
               + (f" The review-text model rates this app's reviews at {r['text_score']:.0%} unverified-like."
                  if r["text_score"] is not None else ""))
    for fr in r["flagged_reviews"]:
        tags = "".join(f'<span class="ls-tag">{html.escape(t)}</span>' for t in fr["tags"])
        when = pd.Timestamp(fr["at"]).strftime("%d %b %Y")
        st.markdown(f'<div class="ls-review">{tags}<span class="ls-muted">{"★" * fr["score"]} · {when}</span>'
                    f'<br>{html.escape(fr["text"][:500])}</div>', unsafe_allow_html=True)
    if not r["flagged_reviews"]:
        st.write("No recent review matched the warning terms.")

st.divider()
st.caption(
    "**This is not a legal determination.** LoanShield is a student project (SIT Pune, TE7483). "
    "It estimates whether an app resembles apps missing from RBI's Digital Lending Apps directory, from "
    "public Play Store data. Always confirm in the "
    f"[official RBI DLA directory]({RBI_DLA_URL}) before sharing documents or taking a loan, and report "
    f"unauthorised lenders on [Sachet]({SACHET_URL}) or at cybercrime.gov.in / 1930.")
with st.expander("Model details"):
    st.json({k: v for k, v in r["card"].items() if k != "tabular_params"} | {
        "raw_model_score": round(r["raw_score"], 4), "calibrated_probability": round(r["probability"], 4),
        "decision_threshold": round(r["threshold"], 4)})

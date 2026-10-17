# Deploying the demo (Streamlit Community Cloud, free)

The repository already contains everything the hosted demo needs:

- `app/streamlit_app.py`, the app (it adds `src/` to the import path itself)
- `app/requirements.txt`, lean pinned dependencies. Community Cloud uses the file next to the entrypoint, so the root `requirements.txt`, which includes CPU-only PyTorch for the MuRIL experiment, is ignored.
- `results/models/`, the trained models (about 5 MB)
- `data/demo/offline_bundle.json.gz`, cached Play data for all 652 apps, so **offline mode works on the cloud**

## Steps (about 5 minutes)

1. Sign in at <https://share.streamlit.io> with the GitHub account that owns `Hardik25-iso/LoanShield`.
2. Click **Create app → Deploy a public app from GitHub**.
3. Repository `Hardik25-iso/LoanShield`, branch `master`, main file path `app/streamlit_app.py`.
4. Under **Advanced settings**, choose **Python 3.12** (the version everything was tested on).
5. Click **Deploy**. The first build takes a few minutes. Share the `*.streamlit.app` link.

Live mode calls Google Play from Streamlit's servers. If Play rate-limits or blocks those requests during the viva, switch the sidebar to **Offline**.

## Updating

Every push to `master` redeploys automatically. After retraining (`notebooks/03_explainability.ipynb`), commit `results/models/`. After re-collecting data, rebuild the offline bundle:

```bash
python -m loanshield.demo --build-offline-bundle
```

# Quant_project

## Automated deployment setup

This repository is configured for a free Render deployment and GitHub CI.

### What was added
- `Procfile` — tells cloud hosts how to start the Streamlit dashboard.
- `render.yaml` — Render service definition for auto-deploy from GitHub.
- `.github/workflows/ci.yml` — automatic CI on `push`/`pull_request` to `main`.

### Deploy to Render
1. Push this repo to GitHub.
2. Sign up at https://render.com and connect your GitHub repository.
3. Create a new Web Service using this repository.
4. Use the default build command and service start command from `render.yaml`.

Render will then auto-deploy on every push to `main`.

### Notes
- The app entrypoint is `dashboard/app.py`.
- Use environment variables for sensitive values instead of committing `.env`.
- `streamlit` is already included in `requirements.txt`.

# Deployment

## GitHub Pages

The `Daily LLM Benchmark Update + Deploy` workflow:

1. Installs cached Python dependencies.
2. Runs unit/fixture tests and runtime lint checks.
3. Scrapes Artificial Analysis and policy-permitted LLMStats HTML.
4. Builds source-native cleaned datasets.
5. Runs identity matching, representatives, consensus, badges, history, and
   quality.
6. Validates publication contracts.
7. Commits only meaningful generated changes.
8. Packages website, required data, and docs.
9. Deploys GitHub Pages.

Manual publication:

```powershell
.\.venv\Scripts\python.exe pipeline\run_pipeline.py --with-sentiment
.\.venv\Scripts\python.exe pipeline\build_family_history.py
.\.venv\Scripts\python.exe pipeline\validate_publication.py
git add .github README.md CHANGELOG.md METHODOLOGY.md api_server.py requirements.txt pipeline scraper tests website docs
git add data/index data/models.json data/models.csv data/raw_snapshot/latest.json data/raw_snapshot/latest.csv
git add data/cleaned data/families data/capabilities data/identity data/methodology data/quality data/history
git commit -m "Add multi-source LLMDEX consensus"
git push
```

Merge to `main` and run the workflow manually if needed.
Timestamped raw source payloads under `data/raw/` are intentionally excluded
from Git history and retained in the workflow's dataset archive artifact.

## Render

`api_server.py` can run on Render for the server-side Advisor and API. Static
GitHub Pages remains sufficient for leaderboards. Render cold starts can delay
the Advisor/API but do not delay static datasets.

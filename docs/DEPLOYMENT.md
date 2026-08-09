# Deployment

## Automated publication

The `Daily LLM Benchmark Update + Deploy` workflow:

1. Installs cached Python dependencies.
2. Runs unit/fixture tests and runtime lint checks.
3. Scrapes Artificial Analysis and policy-permitted LLMStats HTML.
4. Builds source-native cleaned datasets.
5. Runs identity matching, representatives, consensus, badges, history, and
   quality.
6. Validates publication contracts.
7. Commits only meaningful generated changes to `main`.
8. Lets the Render and Cloudflare Pages Git integrations deploy that commit.
9. Triggers the isolated GitHub Pages workflow after a successful publication.

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

## Static hosting

`api_server.py` runs on Render for the server-side Advisor and API. Cloudflare
Pages is the primary static host, while GitHub Pages is an independently built
backup. Render cold starts can delay the Advisor or API but do not delay static
datasets.

The static Cloudflare Pages deployment intentionally uses the deterministic,
dataset-grounded local Advisor unless a separate API origin is configured.
To connect a separately hosted, CORS-enabled backend, define the API origin
before `app.js` loads:

```html
<script>
  window.LLMDEX_API_BASE = "https://your-llmdex-api.example.com";
</script>
```

Alternatively, add a `llmdex-api-base` meta tag with the same origin. The
browser never receives Gemini keys; `api_server.py` reads them server-side.
When the website is served from `localhost`, the frontend automatically uses
the same origin.

The API accepts browser requests from the Render and Cloudflare production
origins and from the same-origin local server by default. Add preview or custom
domains explicitly with a comma-separated environment variable:

```text
LLMDEX_CORS_ALLOWED_ORIGINS=https://preview.example.com,https://llmdex.example.com
```

# Repository Guidelines

## Project Structure & Modules
- `amazon_uk/` Scrapy project: spider at `spiders/amazon_uk.py`, pipelines handle URL normalization/dedup, settings define UA pool/throttling, CSV outputs in `runs/`.
- `api/` FastAPI wrapper (`api/main.py`) with `/interpret`, `/crawl`, `/run`, `/runs`, `/runs-list`, `/health`.
- `ui/` Streamlit client that talks to the API; sidebar controls API base and LLM provider.
- Root: `Dockerfile`, `requirements.txt`, `README.md`, `.env.example` (real `.env` is gitignored).

## Setup, Build & Run
- Python 3.10+ (tested on 3.12). Create venv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install deps: `pip install -r requirements.txt` (Scrapy 2.10.0 + Twisted 22.10.0 pinned for stability).
- CLI crawl: `cd amazon_uk && scrapy crawl amazon_uk -a search="Intel NUC" -a category="computers" -a filter_words="i7,16GB" -a filter_mode="any" -a exception_keywords="refurbished" -o runs/intel_nuc.csv`.
- API: `uvicorn api.main:app --reload --port 8000`; call `/health` then `POST /run` to interpret + crawl.
- UI: `streamlit run ui/app.py --server.port 8501`; set API base + provider in sidebar before running a task.
- Docker (API+UI): `docker build -t amazon-uk-scraper .` then `docker run --env-file .env -p 18000:8000 -p 18501:8501 amazon-uk-scraper`.

## LLM & Environment
- Configure via `.env` (see template): `LLM_BACKEND=ollama|openai`, `LLM_ENDPOINT`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_TIMEOUT`. Keep keys out of git.
- `/run` accepts per-request overrides for backend/endpoint/model/key/timeout when you need a different provider.

## Coding Style & Naming
- PEP 8, 4-space indent; snake_case modules/args; CamelCase classes.
- Keep Scrapy etiquette intact (AutoThrottle, delays, UA rotation). Pipelines normalize `/dp/` and `/gp/product/` links and drop duplicates—maintain that behavior when editing selectors.

## Testing & Validation
- Quick checks: `scrapy check` and a short crawl to confirm selectors, deduped ASINs, and clean links in `runs/`.
- API smoke: `curl -X POST http://localhost:8000/run -H "Content-Type: application/json" -d '{"prompt":"Find Intel NUC i7 16GB","dev_mode":true}'`.
- UI: run against local/remote API; confirm saved runs list populates from `/runs-list`.

## Commit & PR Guidelines
- Use concise, imperative commits (e.g., `Add run summary endpoint`); never commit `.env` or output CSVs.
- PRs should note crawl params used for validation, sample output location, and any LLM/config changes or new env vars.
- Include screenshots/snippets if UI or output shape changes.

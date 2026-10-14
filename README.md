# PetMind

PetMind is a monorepo for a Ukrainian pet food recommendation service. The `src/petrec/` package holds the recommendation method; FastAPI imports and uses that package, and the React client calls the API.

## Prerequisites

- Python 3.11 and Node.js 22+ with pnpm (`corepack enable`).
- PostgreSQL 16 with pgvector — either through Docker Compose or a local install (the Windows setup used for the thesis keeps it in `%LOCALAPPDATA%\PetMind\postgres`).
- Optional: a free Google Gemini API key from <https://aistudio.google.com/apikey>. Without a key the assistant and recommendations run in a deterministic simplified mode.
- Optional research data: `data/processed/products.parquet` and `data/processed/reviews.parquet`. The demo seed works without them.

## Quick start on Windows (local PostgreSQL)

```powershell
Copy-Item .env.example .env        # then set GEMINI_API_KEY if you have one
.\scripts\start-local.ps1
```

The script starts PostgreSQL, applies migrations, runs `seed.py --demo-only`, starts the API on `API_PORT` (default `8000`) and the Vite dev server on <http://localhost:5173>. Vite proxies `/api` to the same `API_PORT`. If an API process started before the latest code changes still holds the port, the script prints a warning with the PID to stop.

## Run with Docker Compose

```powershell
Copy-Item .env.example .env
docker compose up --build -d
docker compose exec backend python scripts/seed.py --demo-only
```

Open the app at <http://localhost:5173>, the API at <http://localhost:8000/api/health>, the LLM state at <http://localhost:8000/api/health/llm> and API docs at <http://localhost:8000/docs>. Volumes: `petly_pgdata` (database), `petly_llm_cache` (LLM response cache), `petmind_hf_cache` (embedding model weights). `docker compose down -v` removes them. The GitHub Actions workflow `.github/workflows/ci.yml` builds and smoke-tests the stack.

## Demo accounts

`seed.py --demo-only` is idempotent and creates 51 clearly labelled demo goods (with ingredient rows, embeddings and knowledge-base chunks), purchase history for reminders and these logins:

| Email | Pet | Purpose |
|---|---|---|
| `murchyk@petmind.local` | Мурчик, sterilized cat, poultry allergy | allergen filtering, reminders |
| `richie@petmind.local` | Річі, dog | dog catalog, reminders |
| `luna@petmind.local` | Луна, cat, no history | cold start |
| `admin@petmind.local` | — | catalog administration (**Адмін**) |

Demo owners use `PETMIND_DEMO_PASSWORD` (default `PetMindDemo2026!`); the administrator uses `PETMIND_ADMIN_PASSWORD` (default `PetMindAdmin2026!`).

## LLM providers

`backend/app/llm/` is a provider layer over the core `petrec.llm.LLMClient`:

- **Gemini** (active by default) talks to Gemini's OpenAI-compatible endpoint through the `openai` package. Set `GEMINI_API_KEY`; models are configured with `LLM_CHAT_MODEL` / `LLM_RERANK_MODEL`.
- **OpenAI, Anthropic, Ollama** are implemented but disabled. To switch, list the provider in `LLM_ENABLED_PROVIDERS`, set `LLM_PROVIDER` and the provider credentials, then restart the API.
- Every client is wrapped in a client-side rate limiter (separate chat and rerank budgets), retries for transient errors and a circuit breaker. When the quota is exhausted or the provider fails, recommendations use the deterministic `s_final` without the LLM term and the chat switches to the simplified mode (`mode: "fallback"`).
- Allergen safety and the compatibility percentage never come from the LLM.

On the free tier Google may use prompts to improve its products; the agent never sends owner names or emails, only pet facts and catalog data.

## Tests

```powershell
python -m pytest -q                                  # unit and API tests (no network)
python -m pytest -q tests/test_live_gemini.py        # live Gemini smoke test; needs GEMINI_API_KEY
python -m pytest -q experiments/e2e/table_3_5_e2e.py # KRM 3.5 acceptance against a running, seeded API
```

The e2e suite signs in as the demo owners automatically. Results are written to `experiments/results/table_3_5.csv`.

## Environment variables

Copy `.env.example` to `.env`; it documents every variable. The most important ones:

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | SQLAlchemy URL for local runs (Compose builds its own) | `postgresql+psycopg://petly:petly@localhost:5432/petly` |
| `API_PORT` | API port used by the backend, Vite proxy and `start-local.ps1` | `8000` |
| `PETMIND_AUTH_SECRET` | Secret that signs login tokens | dev secret; set a random value |
| `PETMIND_ADMIN_EMAIL`, `PETMIND_ADMIN_PASSWORD` | Demo administrator | `admin@petmind.local` / `PetMindAdmin2026!` |
| `PETMIND_DEMO_PASSWORD` | Password of the demo owners | `PetMindDemo2026!` |
| `LLM_PROVIDER` | `gemini`, `openai`, `anthropic`, `ollama` or `disabled` | `gemini` |
| `LLM_ENABLED_PROVIDERS` | Providers allowed to run | `gemini` |
| `GEMINI_API_KEY` | Free Gemini key | empty → simplified mode |
| `LLM_CHAT_MODEL`, `LLM_RERANK_MODEL` | Models for the agent and the rerank | `gemini-flash-lite-latest` |
| `LLM_CHAT_RPM/RPD`, `LLM_RERANK_RPM/RPD` | Client-side request budgets per minute/day | `10/400`, `5/200` |
| `RECOMMENDATION_TTL_SECONDS` | Recommendation cache lifetime (invalidated on profile, purchase and catalog changes) | `600` |
| `PETMIND_API_URL` | Base URL for the live e2e suite | `http://localhost:8000` |

## Project structure and KRM mapping

| Path | Responsibility | KRM subsection |
|---|---|---|
| `src/petrec/` | Ranking, safety, composition scoring, RAG, LLM orchestration, reminders, and recommendation pipeline | 3.3 — recommendation method implementation |
| `backend/app/` | FastAPI routes/schemas, SQLAlchemy persistence, API orchestration; `agent/` (PetMind agent), `llm/` (providers), `safety/` (allergen normalization) | 3.4 — service and API integration |
| `backend/alembic/`, `backend/scripts/seed.py` | PostgreSQL schema migrations and loading processed data/demo pets | 3.4 — data layer and initialization |
| `frontend/src/` | React pages and API-driven PetMind interface | 3.4 — user interface |
| `experiments/` | Evaluation runners and exported result tables | 3.5 — experimental evaluation |
| `experiments/e2e/table_3_5_e2e.py` | HTTP acceptance scenarios and CSV outcomes for the manual-check cases | 3.5 — practical verification |
| `docker-compose.yml` | Local deployment of PostgreSQL/pgvector, API, and web client | 3.4–3.5 — deployment and reproducible verification |

The `src/petrec/` method is the source of truth for recommendation and safety decisions. Backend routes adapt database records and call the package; frontend compatibility badges and percentages are based on deterministic API fields.

## Manual verification checklist (KRM 3.5)

Use a seeded catalog and the demo user/pet records. Confirm each item in the browser and API (`/docs` is available):

- [ ] Select cat Мурчик, whose profile lists a poultry allergy. Request `/api/recommendations?user_id=…&pet_id=…&use_llm=true`; no poultry-containing product appears.
- [ ] In the PetMind AI chat, explicitly ask the assistant to ignore the allergy and recommend poultry food. The assistant refuses that request or returns safe alternatives; no unsafe product card appears.
- [ ] Set `LLM_PROVIDER=disabled` (or remove provider credentials), restart the backend, and request recommendations. A non-empty fallback list is returned and its items report `llm_used=false`.
- [ ] Use a demo account with no purchase history and an existing pet profile. Recommendations still load from the pet profile (cold start).
- [ ] For a pet with an existing replenishment reminder, note `days_left`, record a new purchase for that reminder item through checkout/API, and reload reminders. The forecast is recalculated and `days_left` reflects the updated expected date.

The automated version runs the same four acceptance scenarios through HTTP and records pass/fail plus observed evidence in `experiments/results/table_3_5.csv`. Set `PETLY_USER_ID`/`PETLY_EMAIL`, `PETLY_PET_ID`, `PETLY_COLD_START_USER_ID`/`PETLY_COLD_START_EMAIL`, and `PETLY_COLD_START_PET_ID` when selecting non-default fixture records. The seeded demo records are selected automatically when those values are omitted.

## Experiments (KRM section 4)

Prepare temporal train/test Parquet splits and product/profile Parquet files before running the evaluations:

```powershell
python experiments/run_all.py --train data/processed/train.parquet --test data/processed/test.parquet --products data/processed/products.parquet --profiles data/processed/profiles.parquet --output-dir experiments/results
```

The default `experiments/parameter_grid.csv` contains the documented one-factor sweeps plus a feasible 0.1-step gamma simplex. Pass `--grid-csv path.csv` to use a reviewed alternative. The runner writes E1-E5 CSV/PNG artifacts; E4 records three repetitions per available LLM, latency and JSON-valid-response share. Cost stays blank (`NaN`) because provider token usage/pricing is not exposed uniformly by all clients. The stock-aware reminder metric is only calculated for rows with package energy, package weight and pet weight; other histories are compared using mean-interval and exponential-smoothing forecasts.

This checkout does not include `data/processed/*.parquet`. Until the research data is supplied and the experiment command is run, there are no empirical result tables to report; the demo seed and unit tests are not substitutes for the thesis evaluation sample.

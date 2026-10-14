# PetMind

PetMind is a monorepo for a Ukrainian pet food recommendation service. The `src/petrec/` package holds the recommendation method; FastAPI imports and uses that package, and the React client calls the API.

## Prerequisites

- Docker Desktop with Docker Compose v2.
- Node.js 22+ and pnpm 9+ for local frontend development (`corepack enable` can provide pnpm).
- Python 3.11 and pip for local backend work.
- Prepared Parquet files `data/processed/products.parquet` and `data/processed/reviews.parquet` for loading the catalog and purchase history.
- Optional provider credentials or a locally running Ollama model for LLM explanations. The app works without an LLM using deterministic recommendations.

## Run with Docker Compose

```powershell
Copy-Item .env.example .env
docker compose up --build -d
docker compose ps
docker compose exec backend python scripts/seed.py --demo-only
```

Open the app at <http://localhost:5173>, the API at <http://localhost:8000/api/health>, and interactive API documentation at <http://localhost:8000/docs>. The database is persisted in the `petly_pgdata` volume and LLM response cache in `petly_llm_cache`. Alembic migrations run when the backend container starts. `--demo-only` creates demo profiles, sample feed listings, several brand examples, and veterinary pharmacy cards, so catalog browsing, product details, cart, and checkout can be tried without research Parquet files. Demo details and prices are illustrative; veterinary products are not dispensed by this application. Load the processed catalog for research or real product data. When `products.parquet` and `reviews.parquet` are available, run `docker compose exec backend python scripts/seed.py` to load the catalog, interactions, and embeddings.

To stop the services while preserving data, run `docker compose down`. To also remove stored database/cache data, run `docker compose down -v`.

## Local development

Start PostgreSQL with `docker compose up -d db`, then create and activate a Python 3.11 virtual environment. Install the backend requirements and run the API from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
$env:DATABASE_URL = "postgresql+psycopg://petly:petly@localhost:5432/petly"
python -m alembic -c backend/alembic.ini upgrade head
python backend/scripts/seed.py
uvicorn app.main:app --app-dir backend --reload
```

Without prepared Parquet files, use `python backend/scripts/seed.py --demo-only` after PostgreSQL migrations. This adds demo users and pets, clearly labeled food examples, brand examples, and pharmacy product cards. Food rows without a verified package image use the built-in product-card illustration.

In a second terminal:

```powershell
cd frontend
corepack pnpm install
corepack pnpm dev
```

The Vite development server proxies `/api` to `http://localhost:8001` for the local Windows setup; Docker Compose serves the API on port `8000`. Run unit and API tests from the repository root with `pytest`. The live four-case acceptance suite is run separately with `pytest -q experiments/e2e/table_3_5_e2e.py`; it requires a running API and seeded data. Results are written to `experiments/results/table_3_5.csv`.

The local `--demo-only` seed creates an administrator profile. Sign in at `/auth` with `admin@petmind.local` / `PetMindAdmin2026!`, then open **Адмін** in the navigation to add products or edit their prices, descriptions, images, ingredients, brand, and species. Set `PETMIND_ADMIN_EMAIL` and `PETMIND_ADMIN_PASSWORD` before seeding to use different local credentials.

## Environment variables

Copy `.env.example` to `.env`. Docker Compose reads the file automatically. Local Python processes read `.env` through `python-dotenv` where configured; for local frontend development Vite proxies to the API port.

| Variable | Purpose | Default |
|---|---|---|
| `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | PostgreSQL database and credentials | `petly` |
| `POSTGRES_PORT` | Host port for PostgreSQL | `5432` |
| `API_PORT` | Host port for FastAPI | `8000` |
| `FRONTEND_PORT` | Host port for the web client | `5173` |
| `PETMIND_ADMIN_EMAIL`, `PETMIND_ADMIN_PASSWORD` | Local demo administrator credentials created by `seed.py --demo-only` | `admin@petmind.local` / `PetMindAdmin2026!` |
| `PETMIND_AUTH_SECRET` | Secret used to sign short lived API login tokens | Set a unique random value outside local demos |
| `DATABASE_URL` | SQLAlchemy URL for local/API use; Compose builds the service URL from database variables | Compose internal `db:5432` |
| `LLM_PROVIDER` | `openai`, `anthropic`, `ollama`, or `disabled` | `disabled` in the example; code defaults to `openai` |
| `LLM_MODEL` | Provider model override | Provider default |
| `OPENAI_API_KEY` | OpenAI credential | empty |
| `ANTHROPIC_API_KEY` | Anthropic credential | empty |
| `OLLAMA_HOST` | Ollama server URL | `http://host.docker.internal:11434` in Compose |
| `OLLAMA_MODEL` | Ollama model when `LLM_MODEL` is empty | `llama3.1` |
| `LLM_CACHE_PATH` | Persistent JSON cache path for provider responses | `.cache/petrec_llm_cache.json` locally |
| `PETLY_API_URL` | Base URL for live e2e requests | `http://localhost:8000` |

## Project structure and KRM mapping

| Path | Responsibility | KRM subsection |
|---|---|---|
| `src/petrec/` | Ranking, safety, composition scoring, RAG, LLM orchestration, reminders, and recommendation pipeline | 3.3 — recommendation method implementation |
| `backend/app/` | FastAPI routes/schemas, SQLAlchemy persistence, API orchestration, and agent service | 3.4 — service and API integration |
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

The automated version runs the same four acceptance scenarios through HTTP and records pass/fail plus observed evidence in `experiments/results/table_3_5.csv`. Set `PETLY_USER_ID`, `PETLY_PET_ID`, `PETLY_COLD_START_USER_ID`, and `PETLY_COLD_START_PET_ID` when selecting non-default fixture records. The seeded demo records are selected automatically when those values are omitted.

## Experiments (KRM section 4)

Prepare temporal train/test Parquet splits and product/profile Parquet files before running the evaluations:

```powershell
python experiments/run_all.py --train data/processed/train.parquet --test data/processed/test.parquet --products data/processed/products.parquet --profiles data/processed/profiles.parquet --output-dir experiments/results
```

The default `experiments/parameter_grid.csv` contains the documented one-factor sweeps plus a feasible 0.1-step gamma simplex. Pass `--grid-csv path.csv` to use a reviewed alternative. The runner writes E1-E5 CSV/PNG artifacts; E4 records three repetitions per available LLM, latency and JSON-valid-response share. Cost stays blank (`NaN`) because provider token usage/pricing is not exposed uniformly by all clients. The stock-aware reminder metric is only calculated for rows with package energy, package weight and pet weight; other histories are compared using mean-interval and exponential-smoothing forecasts.

This checkout does not include `data/processed/*.parquet`. Until the research data is supplied and the experiment command is run, there are no empirical result tables to report; the demo seed and unit tests are not substitutes for the thesis evaluation sample.

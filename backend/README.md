# Backend data layer

Install `backend/requirements.txt`, start PostgreSQL with pgvector via `docker compose up -d db`, then run migrations from `backend/` with `alembic upgrade head`. Set `DATABASE_URL` when the database is not the Compose default.

Run the idempotent seed after processed data has been written to `data/processed/`:

```powershell
python backend/scripts/seed.py
```

The seed expects `products.parquet` and `reviews.parquet`. It computes/caches multilingual-e5-small product and knowledge-chunk embeddings; the first run downloads/loads the model. Override the input folder with `--processed-dir`.

The migration integration test uses `TEST_DATABASE_URL` and modifies that database by upgrading its schema. Point it only at a disposable PostgreSQL 16 database with pgvector installed:

```powershell
$env:TEST_DATABASE_URL = "postgresql+psycopg://petly:petly@localhost:5432/petly_test"
python -m pytest tests/test_backend_migration.py -m integration
```

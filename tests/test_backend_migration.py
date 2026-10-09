"""PostgreSQL integration check; set TEST_DATABASE_URL to a disposable pgvector DB."""

import os
import sys
from pathlib import Path

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
pytest.importorskip("alembic")
pytest.importorskip("pgvector")
pytest.importorskip("psycopg")

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))


@pytest.mark.integration
def test_alembic_migration_and_pgvector_cosine_search():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable PostgreSQL 16 + pgvector database")

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, text

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    os.environ["DATABASE_URL"] = database_url
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "alembic"))
    command.upgrade(config, "head")

    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            tables = set(connection.execute(text(
                "SELECT tablename FROM pg_tables WHERE schemaname=current_schema()"
            )).scalars())
            assert {"users", "pets", "products", "product_ingredients", "purchases",
                    "kb_chunks", "reminders", "events"}.issubset(tables)
            assert connection.execute(text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector')")).scalar_one()
            connection.execute(text("CREATE TEMP TABLE vector_probe (id text, embedding vector(384)) ON COMMIT DROP"))
            first = "[" + ",".join(["1"] + ["0"] * 383) + "]"
            second = "[" + ",".join(["0", "1"] + ["0"] * 382) + "]"
            connection.execute(text("INSERT INTO vector_probe VALUES ('near', CAST(:embedding AS vector))"), {"embedding": first})
            connection.execute(text("INSERT INTO vector_probe VALUES ('far', CAST(:embedding AS vector))"), {"embedding": second})
            found = connection.execute(text(
                "SELECT id FROM vector_probe ORDER BY embedding <=> CAST(:query AS vector) LIMIT 1"
            ), {"query": first}).scalar_one()
            assert found == "near"
    finally:
        engine.dispose()

"""Add product stock metadata and replenishment forecast state."""

from alembic import op

revision = "0002_restock_forecasts"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # IF NOT EXISTS also handles databases created from the updated metadata by
    # the initial migration; databases already at revision 0001 get the columns.
    op.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS moisture FLOAT")
    op.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS is_regular BOOLEAN NOT NULL DEFAULT FALSE")
    op.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS package_weight_kg FLOAT")
    op.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS energy_kcal_per_kg FLOAT")
    op.execute("ALTER TABLE purchases ADD COLUMN IF NOT EXISTS quantity INTEGER NOT NULL DEFAULT 1")
    op.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS priority FLOAT NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS status VARCHAR(24) NOT NULL DEFAULT 'new'")
    op.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS reminder_date DATE")
    op.execute("CREATE INDEX IF NOT EXISTS ix_reminders_reminder_date ON reminders (reminder_date)")
    op.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_purchases_quantity_positive') THEN
                ALTER TABLE purchases ADD CONSTRAINT ck_purchases_quantity_positive CHECK (quantity > 0);
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_reminders_status') THEN
                ALTER TABLE reminders ADD CONSTRAINT ck_reminders_status CHECK (status IN ('new','done','dismissed'));
            END IF;
        END $$
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE reminders DROP CONSTRAINT IF EXISTS ck_reminders_status")
    op.execute("ALTER TABLE purchases DROP CONSTRAINT IF EXISTS ck_purchases_quantity_positive")
    op.execute("DROP INDEX IF EXISTS ix_reminders_reminder_date")
    op.execute("ALTER TABLE reminders DROP COLUMN IF EXISTS reminder_date")
    op.execute("ALTER TABLE reminders DROP COLUMN IF EXISTS status")
    op.execute("ALTER TABLE reminders DROP COLUMN IF EXISTS priority")
    op.execute("ALTER TABLE purchases DROP COLUMN IF EXISTS quantity")
    op.execute("ALTER TABLE products DROP COLUMN IF EXISTS energy_kcal_per_kg")
    op.execute("ALTER TABLE products DROP COLUMN IF EXISTS package_weight_kg")
    op.execute("ALTER TABLE products DROP COLUMN IF EXISTS is_regular")
    op.execute("ALTER TABLE products DROP COLUMN IF EXISTS moisture")

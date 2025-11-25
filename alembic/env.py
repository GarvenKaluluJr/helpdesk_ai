# alembic/env.py
import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Add project root to sys.path so we can import backend.app
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, ".."))))

config = context.config

# If alembic.ini has sqlalchemy.url, we can just use that.
# (You already set it to the PostgreSQL URL.)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from helpdesk_ai.backend.app.db import Base  # noqa: E402

target_metadata = Base.metadata


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

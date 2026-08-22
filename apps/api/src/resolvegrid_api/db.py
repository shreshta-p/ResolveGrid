import os
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://resolvegrid:resolvegrid_dev@localhost:5433/resolvegrid",
)

_engine = create_engine(DATABASE_URL)


def get_db() -> Iterator[Session]:
    with Session(_engine) as session:
        yield session

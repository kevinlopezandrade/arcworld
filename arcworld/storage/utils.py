from typing import Optional

from sqlalchemy import MetaData, create_engine, select

from arcworld.internal.constants import Task
from arcworld.storage.fingerprint import decode_normalized_task_sqlite


def get_task(id: str, path: str) -> Optional[Task]:
    """
    Given the 'id' or hash of task, retrieves the task from
    the database stored in 'path'. If no task is found
    returns None

    Args:
        id: The id or hash
        path: Path where the databse is located.

    Returns:
        The task or None.
    """
    engine = create_engine(f"sqlite:///{path}")
    metadata = MetaData()
    metadata.reflect(engine)

    for table in metadata.tables.values():
        with engine.connect() as connection:
            query = select(table).where(table.c.id == id)
            row = connection.execute(query).first()

            if row:
                return decode_normalized_task_sqlite(row.task)
            else:
                return None

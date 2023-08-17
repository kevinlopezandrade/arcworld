from sqlalchemy import MetaData, create_engine, select

from arcworld.storage.fingerprint import decode_normalized_task
from arcworld.utils import plot_task

# Create SQLAlchemy engine
engine = create_engine("sqlite:////Users/kev/tasks.db")

# Reflect the database metadata
metadata = MetaData()
metadata.reflect(engine)

# Get the "gravity" table from the reflected metadata.
gravity_table = metadata.tables["gravity"]

# Create a connection and execute the query
with engine.connect() as connection:
    query = select(gravity_table).where(gravity_table.c.transformation == "Gravitate")
    results = connection.execute(query).fetchall()

# Process the results
for i, result in enumerate(results[:4]):
    task = decode_normalized_task(result.task)
    plot_task(task)
    # print(result.author)

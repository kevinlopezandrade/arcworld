from sqlalchemy import MetaData, create_engine, select

from arcworld.internal.constants import Task
from arcworld.storage.fingerprint import decode_normalized_task
from arcworld.utils import plot_task

# Create SQLAlchemy engine.
# Replace by the path to your copy of the database
engine = create_engine("sqlite://///Users/kev/arcworld/examples/tasks.db")

# Reflect the database metadata.
metadata = MetaData()
metadata.reflect(engine)

# Get the "gravity" table from the reflected metadata.
gravity_table = metadata.tables["gravity"]

# Display the columns of the table.
print(gravity_table.columns.keys())

# Create the query
# Delete the call to 'limit' to get all the rows.
stmt = select(gravity_table).limit(5)

# Only Tasks where some transformation is wanted.
# stmt = select(gravity_table).where(gravity_table.c.transformation == "Gravitate")
# stmt = (
#     select(gravity_table)
#     .where(gravity_table.c.transformation == "DropBidirectionalDots")
#     .limit(5)
# )

# Create a connection and execute the query
with engine.connect() as connection:
    for row in connection.execute(stmt):
        task: Task = decode_normalized_task(row.task)
        plot_task(task)

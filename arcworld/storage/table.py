import tables


class SchemataTaskHDF5(tables.IsDescription):
    """
    Schema for tasks generated from Schematas.

    Args:
        id: Unique hash of the task.
        transformation: Transformation used for the grid.
        author: Email of the author.
        task: Normalized task in a shape of (12,2,30,30)
    """

    id = tables.StringCol(64)
    transformation = tables.StringCol(64)
    author = tables.StringCol(64)
    task = tables.UInt8Col(shape=(12, 2, 30, 30))


class EngineeredTaskHDF5(tables.IsDescription):
    """
    Schema for reverse engineered tasks.

    Args:
        id: Unique hash of the task
        transformation: id of the reverse engineered task found
            at the beggining of the json file.
        split: Set to which it belongs. Either training or evaluation.
        task: Normalized task in a shape of (12,2,30,30)
    """

    id = tables.StringCol(64)
    transformation = tables.StringCol(64)
    split = tables.StringCol(10)
    task = tables.UInt8Col(shape=(12, 2, 30, 30))


class TaskHDF5(tables.IsDescription):
    """
    Schema for the original A.R.C tasks.

    Args:
        id: Unique hash of the task found at the beggining of the json
            file.
        task: Normalized task in a shape of (12,2,30,30)
    """

    id = tables.StringCol(64)
    task = tables.UInt8Col(shape=(12, 2, 30, 30))

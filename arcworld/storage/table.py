import numpy as np
import sqlalchemy.types as types
from numpy.typing import NDArray
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Task(types.TypeDecorator):
    impl = types.LargeBinary

    def process_bind_param(self, value: NDArray[np.uint8], dialect):
        return value.tobytes()

    def process_result_value(self, value: NDArray[np.uint8], dialect):
        return np.frombuffer(value, dtype=np.uint8).reshape(12, 2, 30, 30)


class Base(DeclarativeBase):
    pass


class Schemata(Base):
    """
    Abstract schema for tasks generated from Schematas.

    Args:
        id: Unique hash of the task.
        transformation: Transformation used for the grid.
        author: Email of the author.
        task: Normalized task in a shape of (12,2,30,30)
    """

    __abstract__ = True
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    author: Mapped[str] = mapped_column(String(64))
    transformation: Mapped[str] = mapped_column(String(64))
    task: Mapped[NDArray[np.uint8]] = mapped_column(Task())


class Engineered(Base):
    """
    Schema for reverse engineered tasks.

    Args:
        id: Unique hash of the task
        author: Email of the author.
        transformation: id of the reverse engineered task found
            at the beggining of the json file.
        split: Set to which it belongs. Either training or evaluation.
        task: Normalized task in a shape of (12,2,30,30)
    """

    # __abstract__ = True
    __tablename__ = "reverse_engineered"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    author: Mapped[str] = mapped_column(String(64))
    transformation: Mapped[str] = mapped_column(String(64))
    split: Mapped[str] = mapped_column(String(64))
    task: Mapped[NDArray[np.uint8]] = mapped_column(Task())


class Original(Base):
    """
    Schema for the original A.R.C tasks.

    Args:
        id: Unique hash of the task found at the beggining of the json
            file.
        task: Normalized task in a shape of (12,2,30,30)
    """

    __abstract__ = True
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task: Mapped[NDArray[np.uint8]] = mapped_column(Task())

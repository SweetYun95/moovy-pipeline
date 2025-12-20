# models/popular_movie_snapshot.py
from sqlalchemy import (
    Column,
    Integer,
    String,
    Date,
    DateTime,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .base import Base


class PopularMovieSnapshot(Base):
    __tablename__ = "popular_movie_snapshots"

    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(Date, nullable=False)
    source = Column(String(30), nullable=False)  # ex) 'TMDB_TRENDING_DAILY'
    content_id = Column(Integer, ForeignKey("video_contents.content_id"), nullable=False)
    rank = Column(Integer, nullable=False)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    deleted_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "snapshot_date",
            "source",
            "content_id",
            name="uniq_snapshot_date_source_content_id",
        ),
    )

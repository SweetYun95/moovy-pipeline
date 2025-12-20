# models/video_content.py
from sqlalchemy import Column, Integer, String, Text, Date, DateTime
from sqlalchemy.sql import func

from .base import Base


class VideoContent(Base):
    __tablename__ = "video_contents"

    content_id = Column(Integer, primary_key=True, autoincrement=True)
    tmdb_id = Column(Integer, nullable=False, unique=True)
    title = Column(String(100), nullable=False)
    release_date = Column(Date, nullable=True)
    genre = Column(String(30), nullable=True)
    time = Column(Integer, nullable=True)           # runtime
    age_limit = Column(Integer, nullable=True)
    plot = Column(Text, nullable=True)
    poster_path = Column(String(255), nullable=True)
    backdrop_path = Column(String(255), nullable=True)

    # timestamps + paranoid(true) 대응용 컬럼들
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    deleted_at = Column(DateTime, nullable=True)

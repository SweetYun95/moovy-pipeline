# apps/crawler/popular_snapshot_job.py
from datetime import date, datetime
from typing import Iterable

from sqlalchemy.orm import Session

from configs.settings import SNAPSHOT_LIMIT, SNAPSHOT_SOURCE
from models.base import SessionLocal
from models.video_content import VideoContent
from models.popular_movie_snapshot import PopularMovieSnapshot
from .tmdb_client import TMDBClient


def _parse_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def upsert_video_content(session: Session, movie: dict) -> VideoContent:
    """
    TMDB movie JSON → VideoContent upsert
    movie: 하나의 TMDB 영화 객체
    """
    tmdb_id = movie["id"]
    vc: VideoContent | None = (
        session.query(VideoContent).filter_by(tmdb_id=tmdb_id).one_or_none()
    )

    release_date = _parse_date(movie.get("release_date"))
    # 장르는 일단 genre_ids를 문자열로
    genre_ids = movie.get("genre_ids") or []
    genre = ", ".join(str(g) for g in genre_ids) if genre_ids else None

    base_fields = {
        "title": movie.get("title") or movie.get("original_title"),
        "release_date": release_date,
        "genre": genre,
        "time": None,  # detail API로 runtime 채울 생각이면 나중에 추가
        "age_limit": None,  # 등급 정보도 나중에 확장
        "plot": movie.get("overview"),
        "poster_path": movie.get("poster_path"),
        "backdrop_path": movie.get("backdrop_path"),
    }

    if vc is None:
        vc = VideoContent(tmdb_id=tmdb_id, **base_fields)
        session.add(vc)
        session.flush()  # content_id 채우기
    else:
        for k, v in base_fields.items():
            setattr(vc, k, v)

    return vc


def save_popular_snapshot_for_today(limit: int | None = None):
    """
    1) TMDB trending movie/day 가져온 뒤
    2) video_contents upsert
    3) popular_movie_snapshots에 오늘자 스냅샷 저장
    """
    client = TMDBClient()
    movies = client.get_trending_movies(time_window="day")
    if limit:
        movies = movies[:limit]

    snapshot_date = date.today()
    source = SNAPSHOT_SOURCE

    with SessionLocal() as session:
        try:
            # 오늘자 + source 기존 스냅샷 삭제(soft delete 아니고 실제 delete)
            session.query(PopularMovieSnapshot).filter_by(
                snapshot_date=snapshot_date, source=source
            ).delete()

            snapshots: list[PopularMovieSnapshot] = []

            for rank, movie in enumerate(movies, start=1):
                vc = upsert_video_content(session, movie)
                snapshot = PopularMovieSnapshot(
                    snapshot_date=snapshot_date,
                    source=source,
                    content_id=vc.content_id,
                    rank=rank,
                )
                session.add(snapshot)
                snapshots.append(snapshot)

            session.commit()
            print(
                f"[OK] snapshot_date={snapshot_date}, "
                f"source={source}, count={len(snapshots)}"
            )
        except Exception as e:
            session.rollback()
            print("[ERROR] failed to save snapshot:", e)
            raise

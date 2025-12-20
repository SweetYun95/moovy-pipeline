# configs/settings.py
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# 루트에서 .env 읽기
load_dotenv()

@dataclass
class DBConfig:
    host: str = os.getenv("MOOVY_DB_HOST", "localhost")
    port: int = int(os.getenv("MOOVY_DB_PORT", "3306"))
    user: str = os.getenv("MOOVY_DB_USER", "root")
    password: str = os.getenv("MOOVY_DB_PASSWORD", "")
    name: str = os.getenv("MOOVY_DB_NAME", "moovy")

    @property
    def url(self) -> str:
        # pymysql 기준
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}?charset=utf8mb4"
        )


@dataclass
class TMDBConfig:
    api_key: str = os.getenv("TMDB_API_KEY", "")
    base_url: str = "https://api.themoviedb.org/3"
    language: str = os.getenv("TMDB_LANGUAGE", "ko-KR")


DB_CONFIG = DBConfig()
TMDB_CONFIG = TMDBConfig()
SNAPSHOT_SOURCE = os.getenv("SNAPSHOT_SOURCE", "TMDB_TRENDING_DAILY")
SNAPSHOT_LIMIT = int(os.getenv("SNAPSHOT_LIMIT", "20"))

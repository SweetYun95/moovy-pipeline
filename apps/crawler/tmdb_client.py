# apps/crawler/tmdb_client.py
import requests
from typing import List, Dict

from configs.settings import TMDB_CONFIG


class TMDBClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or TMDB_CONFIG.api_key
        if not self.api_key:
            raise ValueError("TMDB_API_KEY is not set")
        self.base_url = TMDB_CONFIG.base_url
        self.language = TMDB_CONFIG.language

    def _get(self, path: str, params: Dict | None = None) -> Dict:
        url = f"{self.base_url}{path}"
        params = params or {}
        params["api_key"] = self.api_key
        params["language"] = self.language

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_trending_movies(self, time_window: str = "day") -> List[Dict]:
        """
        https://api.themoviedb.org/3/trending/movie/day
        """
        data = self._get(f"/trending/movie/{time_window}")
        return data.get("results", [])

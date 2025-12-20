# scripts/run_trending_snapshot.py
from apps.crawler.popular_snapshot_job import save_popular_snapshot_for_today
from configs.settings import SNAPSHOT_LIMIT

if __name__ == "__main__":
    save_popular_snapshot_for_today(limit=SNAPSHOT_LIMIT)

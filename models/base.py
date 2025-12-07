# models/base.py
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from configs.settings import DB_CONFIG

engine = create_engine(
    DB_CONFIG.url,
    echo=False,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()

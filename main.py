# RE-Earth_recommend/main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from sqlalchemy import create_engine
from apscheduler.schedulers.background import BackgroundScheduler

# 전역변수
model = None
df = None
user_en = LabelEncoder()
action_en = LabelEncoder()

# .env 파일 로드
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# SQLAlchemy로 MySQL 연결
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# 유저의 친환경 활동 별 활동량 데이터 집계
query = """
SELECT userId, ecoActionId, SUM(quantity) as total_quantity
FROM ecoActionLogs
GROUP BY userId, ecoActionId;
"""


app = FastAPI()

# 허용할 origin 설정
origins = [
    os.getenv("FRONTEND_APP_URL"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # 허용할 origin
    allow_credentials=True,         # 쿠키 인증 허용 여부
    allow_methods=["*"],            # 허용할 HTTP 메서드 (GET, POST 등)
    allow_headers=["*"],            # 허용할 HTTP 헤더
)


def train_model():
    global model, df, user_en, action_en
    
    query = """
    SELECT userId, ecoActionId, SUM(quantity) as total_quantity
    FROM ecoActionLogs
    GROUP BY userId, ecoActionId;
    """
    data = pd.read_sql(query, engine)
    df = data[['userId','ecoActionId','total_quantity']].copy()

    df['user_id_enc'] = user_en.fit_transform(df['userId'])
    df['action_id_enc'] = action_en.fit_transform(df['ecoActionId'])

    matrix = csr_matrix(
        (df["total_quantity"], (df["user_id_enc"], df["action_id_enc"]))
    )

    model = AlternatingLeastSquares(factors=10, iterations=50)
    model.fit(matrix.T)
    print("✅ ALS 모델 재학습 완료")

@app.on_event("startup")
def startup_event():
    # 앱 시작 시 최초 학습
    train_model()

    # APScheduler 실행 (24시간마다 재학습)
    scheduler = BackgroundScheduler()
    scheduler.add_job(train_model, "interval", hours=24)
    scheduler.start()

data = pd.read_sql(query, engine)

# 필요한 값 추출 -> 데이터프레임으로 변환
df=data[['userId','ecoActionId','total_quantity']]



user_en=LabelEncoder()
action_en=LabelEncoder()

# 유저 아이디, 환경 아이디 인코딩(고유값)
df['user_id_enc'] = user_en.fit_transform(df['userId'])
df['action_id_enc'] = action_en.fit_transform(df['ecoActionId'])

# 인코딩 좌표에 활동량 값을 포함하여 매트릭스 생성
# 유저-활동량 행렬로 학습
matrix = csr_matrix(
    (df["total_quantity"], (df["user_id_enc"], df["action_id_enc"]))
)
model = AlternatingLeastSquares(factors=10, iterations=50)
model.fit(matrix.T)




@app.get("/action_stats")
def action_stats( user_id: int = Query(..., description="원본 user_id 입력"),
    top_n: int = 5):
    if user_id not in df['userId'].values:
        raise HTTPException(status_code=404, detail="해당 user_id는 데이터에 없습니다.")
    # 활동량이 가장 많은 ecoaction
    user_actions = (
        df[df['userId'] == user_id]
        .groupby('ecoActionId')['total_quantity']
        .sum()
        .reset_index()
    )
    top_action = user_actions.loc[user_actions['total_quantity'].idxmax()]
    top_action_id = int(top_action['ecoActionId'])
    user_value = float(top_action['total_quantity'])

    user_idx = user_en.transform([user_id])[0]
    similar_users = model.similar_users(user_idx, N=top_n)
    similar_user_ids = [
    user_en.inverse_transform([u])[0] 
    for u, _ in similar_users if u != user_idx
]
    
    avg_value = (
        df[(df['userId'].isin(similar_user_ids)) & (df['ecoActionId'] == top_action_id)]
        ['total_quantity']
        .mean()
    )

    return {
        "user_id": user_id,
        "top_action_id": top_action_id,
        "user_value": user_value,
        "similar_user_avg": float(avg_value) if not pd.isna(avg_value) else 0,
        "similar_users": similar_user_ids,
    }


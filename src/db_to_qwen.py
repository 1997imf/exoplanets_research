
import json

import pandas as pd
import requests
from sqlalchemy import create_engine, text


# ==========================================
# 1. PostgreSQL 연결 설정
# ==========================================

DB_URL = (
    "postgresql+psycopg://postgres:postgres"
    "@localhost:5432/exoplanets"
)

engine = create_engine(DB_URL)


# ==========================================
# 2. SQL로 분석할 행성 데이터 추출
# ==========================================

query = text("""
    SELECT
        pl_name,
        pl_rade,
        pl_bmasse,
        pl_orbper,
        discoverymethod
    FROM planetary_systems
    WHERE pl_rade < 2
      AND pl_rade IS NOT NULL
    ORDER BY pl_rade
    LIMIT 20;
""")

try:
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
finally:
    engine.dispose()

if df.empty:
    raise SystemExit("No exoplanet data found.")

print("=== Data from PostgreSQL ===")
print(df.to_string(index=False))


# ==========================================
# 3. Python으로 데이터 정리 및 통계 계산
# ==========================================

numeric_columns = [
    "pl_rade",
    "pl_bmasse",
    "pl_orbper",
]

# 숫자형 컬럼에 숫자가 아닌 값이 있을 경우 결측치로 처리
for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

stats = {
    "row_count": int(len(df)),
    "discovery_method_counts": {
        str(method): int(count)
        for method, count in (
            df["discoverymethod"]
            .fillna("Unknown")
            .value_counts()
            .items()
        )
    },
    "numeric_summary": {},
}

# 반지름, 질량, 공전 주기의 요약 통계 계산
for column in numeric_columns:
    values = df[column].dropna()

    if values.empty:
        stats["numeric_summary"][column] = {
            "valid_count": 0
        }
        continue

    stats["numeric_summary"][column] = {
        "valid_count": int(values.count()),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
        "mean": round(float(values.mean()), 4),
        "median": round(float(values.median()), 4),
    }

# 공전 주기가 3일 미만인 행성의 비율
periods = df["pl_orbper"].dropna()

under_3_days = int((periods < 3).sum())
valid_period_count = int(periods.count())

stats["orbital_period_under_3_days"] = {
    "count": under_3_days,
    "valid_count": valid_period_count,
    "percentage": (
        round(under_3_days / valid_period_count * 100, 1)
        if valid_period_count > 0
        else None
    ),
}

print("\n=== Statistics Calculated by Python ===")
print(json.dumps(stats, indent=2, ensure_ascii=False))


# ==========================================
# 4. 데이터와 통계를 Qwen에 전달할 프롬프트 작성
# ==========================================

stats_text = json.dumps(
    stats,
    indent=2,
    ensure_ascii=False
)

data_text = df.to_string(index=False)

prompt = f"""
You are a scientific data analyst.

Analyze the exoplanet data retrieved from a PostgreSQL
database containing data from the NASA Exoplanet Archive.

The units are:
- pl_rade: Earth radii
- pl_bmasse: Earth masses
- pl_orbper: orbital period in days

IMPORTANT RULES:
1. The Python-computed statistics below are authoritative.
2. Use the provided row count and method counts exactly.
3. Do not invent numbers or claim to have data that is absent.
4. Clearly distinguish observed patterns from hypotheses.
5. Do not infer density or planetary composition as a fact
   from the provided table alone.
6. This is a selected sample, not the entire exoplanet archive.

PYTHON-COMPUTED STATISTICS:
{stats_text}

DATA ROWS:
{data_text}

Write a concise report in 200-300 words with these sections:

1. Dataset overview
2. Patterns in radius, mass, and orbital period
3. Discovery methods
4. Unusual values to verify
5. Limitations

Use the Python-computed statistics exactly.
Do not invent facts.
Keep the entire report concise.
"""


# ==========================================
# 5. Docker Model Runner를 통해 Qwen 추론 실행
# ==========================================

MODEL = (
    "huggingface.co/lmstudio-community/"
    "qwen3.5-9b-gguf:Q4_K_M"
)

API_URL = "http://localhost:12434/engines/v1/chat/completions"

try:
    response = requests.post(
        API_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
            "chat_template_kwargs": {
                "enable_thinking": False
            },
        },
        timeout=300,
    )
except requests.RequestException as error:
    raise SystemExit(f"Could not reach Qwen API: {error}")

if not response.ok:
    print("Qwen request failed:")
    print("Status:", response.status_code)
    print(response.text)
    raise SystemExit(1)

result = response.json()
choices = result.get("choices", [])

if not choices:
    print("Unexpected Qwen response:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(1)

choice = choices[0]
answer = choice.get("message", {}).get("content", "")

print("\n=== Qwen Analysis ===")

if answer and answer.strip():
    print(answer.strip())
else:
    print("Qwen returned an empty answer.")
    print("Finish reason:", choice.get("finish_reason"))
    print(json.dumps(result, indent=2, ensure_ascii=False))
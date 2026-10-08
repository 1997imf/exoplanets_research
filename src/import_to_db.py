import pandas as pd
from sqlalchemy import create_engine

csv_file = "../data/processed/pscomppars_clean.csv"

df = pd.read_csv(csv_file)

engine = create_engine(
    "postgresql+psycopg://postgres:postgres@localhost:5432/exoplanets"
)

df.to_sql(
    "planetary_systems",
    engine,
    if_exists="replace",
    index=False
)

print(f"Imported {len(df)} rows and {len(df.columns)} columns.")
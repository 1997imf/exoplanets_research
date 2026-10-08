import pandas as pd

input_file = "../data/raw/PSCompPars_01.csv"
output_file = "../data/processed/pscomppars_clean.csv"

df = pd.read_csv(input_file, comment="#")

df.to_csv(output_file, index=False)

print(f"Saved {len(df)} rows and {len(df.columns)} columns.")
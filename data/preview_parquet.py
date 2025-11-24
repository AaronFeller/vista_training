import pandas as pd
import sys

file = sys.argv[0]

print(f"Loading: {file}")
df = pd.read_parquet(file)

print("\nFirst 5 rows:")
print(df.head())
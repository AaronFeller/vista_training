import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Usage: python preview_parquet.py <file.parquet>")
    sys.exit(1)

file = sys.argv[1]

print(f"Loading: {file}")
df = pd.read_parquet(file)

print("\nFirst 5 rows:")
print(df.head())
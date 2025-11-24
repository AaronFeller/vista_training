import pandas as pd
import sys

file = sys.argv[1]

print(f"Loading: {file}")
# if file ends with .parquet
if file.endswith('.parquet'):
    df = pd.read_parquet(file)
elif file.endswith('.csv'):
    df = pd.read_csv(file)
else:
    print("Unsupported file format. Please provide a .parquet or .csv file.")
    sys.exit(1)
# calculate the longest "sequence" in the dataframe
max_len = df['sequence'].str.len().max()
print(f"Max sequence length: {max_len}")
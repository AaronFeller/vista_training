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
# print head
print("Dataframe head:")
print(df.head())
max_len = df['smiles'].apply(len).max()
print(f"Number of entries: {len(df)}")
print(f"Longest sequence length: {max_len}")
print(f"Columns: {df.columns.tolist()}")
# print basic stats about lengths
lengths = df['smiles'].apply(len)
print("Sequence length statistics:")
print(f"  Mean: {lengths.mean()}")
print(f"  Median: {lengths.median()}")
print(f"  Std: {lengths.std()}")
print(f"  Max: {lengths.max()}")
print(f"  Min: {lengths.min()}")
# print length distribution

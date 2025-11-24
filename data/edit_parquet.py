import pandas as pd
import sys

# for file in dir(sys):
dir = sys.argv[1]
# find all parquet files in the given directory
import os

parquet_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.parquet')]
for file in parquet_files:
    # load the file
    df = pd.read_parquet(file)
    # remove all rows where sequence is > 300 characters
    df = df[df['sequence'].str.len() <= 300]
    # write out to new csv file
    out_file = file.replace('.parquet', '_filtered.csv')
    df.to_csv(out_file, index=False)
    print(f"Processed {file}, saved to {out_file}")
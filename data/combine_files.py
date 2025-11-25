# combine all filtered csv files into one
import os
import sys
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    input_dir = sys.argv[1]
    # pull last layer directory name
    last_layer_dir = os.path.basename(os.path.normpath(input_dir))
    output_file = os.path.join(input_dir, f"{last_layer_dir}_smiles.parquet")

    csv_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith("filtered_with_smiles.csv")
    ]

    combined_df = pd.DataFrame()
    for f in tqdm(csv_files, desc="Combining files"):
        df = pd.read_csv(f)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # output as parquet for faster loading
    combined_df.to_parquet(output_file, index=False)
    print(f"Combined {len(csv_files)} files into {output_file}")
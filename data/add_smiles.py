import pandas as pd
import os, sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

AMINO_TO_SMILES = {
    "A": "N[C@@H](C)C(=O)O",
    "R": "N[C@@H](CCCNC(=N)N)C(=O)O",
    "N": "N[C@@H](CC(N)=O)C(=O)O",
    "D": "N[C@@H](CC(=O)O)C(=O)O",
    "C": "N[C@@H](CS)C(=O)O",
    "E": "N[C@@H](CCC(=O)O)C(=O)O",
    "Q": "N[C@@H](CCC(N)=O)C(=O)O",
    "G": "NCC(=O)O",
    "H": "N[C@@H](CC1=CN=CN1)C(=O)O",
    "I": "N[C@@H]([C@H](CC)C)C(=O)O",
    "L": "N[C@@H](CC(C)C)C(=O)O",
    "K": "N[C@@H](CCCCN)C(=O)O",
    "M": "N[C@@H](CCSC)C(=O)O",
    "F": "N[C@@H](CC1=CC=CC=C1)C(=O)O",
    "P": "N1[C@@H](CCC1)C(=O)O",
    "S": "N[C@@H](CO)C(=O)O",
    "T": "N[C@@H]([C@H](O)C)C(=O)O",
    "W": "N[C@@H](CC(=CN2)C1=C2C=CC=C1)C(=O)O",
    "Y": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "V": "N[C@@H](C(C)C)C(=O)O",
}

def p2smi(seq):
    return "".join(AMINO_TO_SMILES.get(ch, "") for ch in seq)+"H"


def process_file(path):
    df = pd.read_csv(path)

    if "smiles" in df.columns:
        return path

    seqs = df["sequence"].tolist()

    # multiprocessing at sequence level
    with Pool(cpu_count()) as p:
        smiles = list(tqdm(
            p.imap(p2smi, seqs, chunksize=2000),
            total=len(seqs),
            desc=f"Converting {os.path.basename(path)}"
        ))

    df["smiles"] = smiles
    # out_path
    out_path = path.replace(".csv", "_with_smiles.csv")
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    input_dir = sys.argv[1]

    csv_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".csv")
    ]

    for f in tqdm(csv_files, desc="Processing files"):
        process_file(f)
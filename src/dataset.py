from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import numpy as np
from rdkit import Chem

class SequenceDataset(Dataset):
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)
        self.seqs = self.df["smiles"].tolist()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        # return SMILES string only
        return self.seqs[idx]


def apply_span_masking(
    input_ids,
    attention_mask,
    pad_token_id,
    mask_token_id,
    mean_span=10,
    p_mask=0.25,
    poisson_sd=3.0,
):
    B, T = input_ids.shape

    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)

    attn_np = attention_mask.cpu().numpy()

    for i in range(B):
        length = int(attn_np[i].sum())
        if length == 0:
            continue

        n_to_mask = max(1, int(p_mask * length))
        masked = 0

        while masked < n_to_mask:
            raw = np.random.normal(loc=mean_span, scale=poisson_sd)
            span_len = max(1, int(raw))

            start = np.random.randint(0, max(1, length - span_len))
            end = min(start + span_len, length)

            masked_input_ids[i, start:end] = mask_token_id
            labels[i, start:end] = input_ids[i, start:end]
            masked += (end - start)

    return masked_input_ids, labels


def collate_fn(batch, pad_token_id, mask_token_id, tokenizer):
    # ----------------------------------------
    # RDKit randomization
    # ----------------------------------------
    randomized = []
    for smi in batch:
        try:
            mol = Chem.MolFromSmiles(smi)
            smi2 = Chem.MolToSmiles(mol, isomericSmiles=True, doRandom=True)
            randomized.append(smi2)
        except:
            randomized.append(smi)

    # ----------------------------------------
    # Tokenize entire batch at once
    # ----------------------------------------
    tokens = tokenizer(
        randomized,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=2048,
    )

    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # ----------------------------------------
    # Apply span masking
    # ----------------------------------------
    masked_ids, labels = apply_span_masking(
        input_ids,
        attention_mask,
        pad_token_id,
        mask_token_id,
    )

    return {
        "input_ids": masked_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
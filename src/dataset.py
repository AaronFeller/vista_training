from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from torch.utils.data import DataLoader
import lightning as L

class MLMDataModule(L.LightningDataModule):
    def __init__(self, train_path, valid_path, tokenizer, batch_size, mask_token_id, pad_token_id):
        super().__init__()
        self.train_path = train_path
        self.valid_path = valid_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

    def setup(self, stage=None):
        self.train_dataset = SequenceDataset(self.train_path)
        self.valid_dataset = SequenceDataset(self.valid_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda b: collate_fn(
                b,
                self.pad_token_id,
                self.mask_token_id,
                self.tokenizer,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size * 4,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda b: collate_fn(
                b,
                self.pad_token_id,
                self.mask_token_id,
                self.tokenizer,
            ),
        )

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


def random_token_truncation(input_ids, attention_mask, max_length=512):
    length = input_ids.shape[0]

    if length <= max_length:
        return input_ids, attention_mask

    # choose random section to use
    start_idx = np.random.randint(0, length - max_length + 1)
    input_ids = input_ids[start_idx : start_idx + max_length]
    attention_mask = attention_mask[start_idx : start_idx + max_length]

    return input_ids, attention_mask

def collate_fn(batch, pad_token_id, mask_token_id, tokenizer, max_length=512):
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
        padding=False,
        truncation=False,
        add_special_tokens=True,
        max_length=204800,  # effectively no truncation at this step
    )

    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # ----------------------------------------
    # Random truncation to max_length (512)
    # ----------------------------------------
    input_ids, attention_mask = random_token_truncation(input_ids, attention_mask, max_length=max_length)

    # ----------------------------------------
    # pad to max_length
    # ----------------------------------------
    pad_len = max_length - input_ids.shape[0]
    if pad_len > 0:
        input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id)])
        attention_mask = torch.cat([attention_mask, torch.zeros(pad_len)])

    # ----------------------------------------
    # Apply span masking
    # ----------------------------------------
    masked_ids, labels = apply_span_masking(
        input_ids,
        attention_mask,
        pad_token_id,
        mask_token_id,
        max_length=max_length,
    )

    return {
        "input_ids": masked_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
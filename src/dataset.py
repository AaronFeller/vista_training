from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import numpy as np


class SequenceDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name):
        self.df = pd.read_csv(csv_path)
        self.seqs = self.df["sequence"].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        tokens = self.tokenizer(
            seq,
            padding=False,
            truncation=True,
            max_length=2048,
            add_special_tokens=True,
        )
        return {
            "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokens["attention_mask"], dtype=torch.long),
        }


def apply_span_masking(
    input_ids,
    attention_mask,
    pad_token_id,
    mask_token_id,
    mean_span=10,
    p_mask=0.25,
    poisson_sd=3.0
):
    """
    input_ids:     (B, T)
    attention_mask:(B, T)
    """

    B, T = input_ids.shape

    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)

    ids_np = input_ids.cpu().numpy()
    attn_np = attention_mask.cpu().numpy()

    for i in range(B):
        length = int(attn_np[i].sum())  # valid tokens only
        if length == 0:
            continue

        n_to_mask = max(1, int(p_mask * length))
        masked_count = 0

        while masked_count < n_to_mask:
            # Poisson-like span length
            raw = np.random.normal(loc=mean_span, scale=poisson_sd)
            span_len = max(1, int(raw))

            # sample start position
            start = np.random.randint(0, max(1, length - span_len))
            end = min(start + span_len, length)

            masked_input_ids[i, start:end] = mask_token_id
            labels[i, start:end] = input_ids[i, start:end]

            masked_count += (end - start)

    return masked_input_ids, labels


def collate_fn(batch, pad_token_id, mask_token_id):
    input_ids = [b["input_ids"] for b in batch]
    attn_masks = [b["attention_mask"] for b in batch]

    # Dynamic padding
    padded_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id
    )
    padded_masks = pad_sequence(
        attn_masks,
        batch_first=True,
        padding_value=0
    )

    masked_input_ids, labels = apply_span_masking(
        padded_ids,
        padded_masks,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        mean_span=10,
        p_mask=0.25,
        poisson_sd=3.0
    )

    return {
        "input_ids": masked_input_ids,
        "attention_mask": padded_masks,
        "labels": labels,
    }
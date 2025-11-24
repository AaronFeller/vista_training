from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd

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
            max_length=2048,   # hard cap if you want
            add_special_tokens=True,
        )
        return {
            "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokens["attention_mask"], dtype=torch.long),
        }

def collate_fn(batch, pad_token_id):
    input_ids = [b["input_ids"] for b in batch]
    attn_masks = [b["attention_mask"] for b in batch]

    # pad to max length in *this batch*
    padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    return {
        "input_ids": padded_ids,
        "attention_mask": padded_masks,
    }
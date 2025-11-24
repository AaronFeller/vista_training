# src/train_mlm.py
import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer
from dataset import SequenceDataset, collate_fn
from model import MLM_model, model_config

def init_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1
    return rank, world_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="aaronfeller/PeptideMTR_sm")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    rank, world_size = init_distributed()
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id

    dataset = SequenceDataset(args.train_csv, args.tokenizer_name)

    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
    )

    config = model_config(
        vocab_size=tokenizer.vocab_size,
        # fill in rest from your config
    )
    model = MLM_model(config).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=ids, attention_mask=mask)
            logits = outputs["logits"]

            # build your MLM loss here (masked LM, etc.)
            loss = compute_mlm_loss(logits, labels, pad_token_id)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if rank == 0:
            # save checkpoint
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"epoch_{epoch}.pt"))

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
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
import torch.nn.functional as F

def safe_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_checkpoint(rank, model, optimizer, epoch, step, output_dir, keep_last=5):
    if rank != 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    module = model.module if hasattr(model, "module") else model

    state = {
        "epoch": epoch,
        "step": step,
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    # 1. rolling checkpoint
    safe_save(state, os.path.join(output_dir, "latest.pt"))

    # 2. epoch checkpoint
    ckpt_name = f"epoch_{epoch:05d}.pt"
    ckpt_path = os.path.join(output_dir, ckpt_name)
    safe_save(state, ckpt_path)

    # 3. prune older checkpoints
    ckpts = sorted(
        [c for c in os.listdir(output_dir) if c.startswith("epoch_") and c.endswith(".pt")]
    )
    if len(ckpts) > keep_last:
        to_delete = ckpts[: len(ckpts) - keep_last]
        for fname in to_delete:
            os.remove(os.path.join(output_dir, fname))

def load_checkpoint(path, model, optimizer, device):
    if not os.path.exists(path):
        return 0, 0

    ckpt = torch.load(path, map_location=device)

    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    start_epoch = ckpt.get("epoch", 0)
    step = ckpt.get("step", 0)

    print(f"Restored from {path} at epoch {start_epoch}, step {step}")
    return start_epoch, step

def compute_mlm_loss(logits, labels, pad_token_id):
    vocab = logits.size(-1)
    return F.cross_entropy(
        logits.view(-1, vocab),
        labels.view(-1),
        ignore_index=-100
    )

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
    parser.add_argument("--valid_csv", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="aaronfeller/PeptideMTR_sm")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    rank, world_size = init_distributed()
    device = torch.device("cuda", rank)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id

    dataset = SequenceDataset(args.train_csv, args.tokenizer_name)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if world_size > 1 else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, pad_token_id, mask_token_id),
    )

    config = model_config(
        vocab_size=tokenizer.vocab_size,
        # other params filled by config defaults or your settings
    )
    model = MLM_model(config).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch, global_step = load_checkpoint(
        os.path.join(args.output_dir, "latest.pt"),
        model,
        optimizer,
        device
    )

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()

        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, mask=mask)

            loss = compute_mlm_loss(logits, labels, pad_token_id)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # Optional: save mid-epoch (e.g., every 1000 steps)
            if global_step % 10000 == 0:
                save_checkpoint(rank, model, optimizer, epoch, global_step, args.output_dir)

        # Save at end of epoch
        save_checkpoint(rank, model, optimizer, epoch + 1, global_step, args.output_dir)

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
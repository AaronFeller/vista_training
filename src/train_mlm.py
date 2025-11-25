# src/train_mlm.py
print("Starting training script...")

import os
import time
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer
from dataset import SequenceDataset, collate_fn
from model import MLM_model, model_config
import torch.nn.functional as F

# set PYTORCH_CUDA_ALLOC_CONF to max_split_size_mb:128 to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

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
        print(f"No checkpoint found at {path}, starting from scratch.")
        return 0, 0

    ckpt = torch.load(path, map_location=device)

    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    start_epoch = ckpt.get("epoch", 0)
    step = ckpt.get("step", 0)

    print(f"Restored from {path} at epoch {start_epoch}, step {step}")
    return start_epoch, step

def compute_mlm_loss(logits, labels):
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
    parser.add_argument("--train_parquet", type=str, required=True)
    parser.add_argument("--valid_parquet", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="aaronfeller/PeptideMTR_sm")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    last_save_time = time.time()

    rank, world_size = init_distributed()
    device = torch.device("cuda:0")
    # print once time from first rank
    if rank == 0:
        print(f"Starting training at {last_save_time}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id

    # ---- Datasets ----
    if rank == 0:
        print("Loading datasets...")
    train_dataset = SequenceDataset(args.train_parquet)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, pad_token_id, mask_token_id, tokenizer),
    )

    valid_dataset = SequenceDataset(args.valid_parquet)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id, mask_token_id, tokenizer),
    )

    # ---- Model ----
    if rank == 0:
        print("Initializing model...")
    config = model_config(vocab_size=tokenizer.vocab_size)
    model = MLM_model(config).to(device)

    # print number of parameters
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {num_params/1e6:.2f} million parameters.")

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch, global_step = load_checkpoint(
        os.path.join(args.output_dir, "latest.pt"),
        model,
        optimizer,
        device
    )

    # create file for logging if rank 0
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, "training_log.txt")
        with open(log_path, "a") as f:
            f.write("epoch,step,train_loss,valid_loss\n")

    # ============================================================
    #                     TRAINING LOOP
    # ============================================================
    if rank == 0:
        print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()

        num_batches = len(train_loader)
        validate_every = max(1, num_batches // 5)   # 20% of the epoch

        for batch_idx, batch in enumerate(train_loader):

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=ids, attention_mask=mask)
            loss = compute_mlm_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # ---- Training log every 10 batches ----
            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch {epoch} | Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")
                with open(log_path, "a") as f:
                    f.write(f"{epoch},{global_step},{loss.item()},\n")

            # ====================================================
            #       VALIDATION + CHECKPOINT every 20%
            # ====================================================
            if (batch_idx + 1) % validate_every == 0:
                # sync all ranks before validation
                if world_size > 1:
                    dist.barrier()

                if rank == 0:
                    model.eval()
                    val_losses = []

                    with torch.no_grad():
                        # only validate on up to 100 batches to save time
                        for i, vbatch in enumerate(valid_loader):
                            if i >= 100:
                                break
                            vids = vbatch["input_ids"].to(device)
                            vmask = vbatch["attention_mask"].to(device)
                            vlabels = vbatch["labels"].to(device)

                            vlogits = model(input_ids=vids, attention_mask=vmask)
                            vloss = compute_mlm_loss(vlogits, vlabels)
                            val_losses.append(vloss.item())

                    mean_vloss = sum(val_losses) / len(val_losses)

                    with open(log_path, "a") as f:
                        f.write(f"{epoch},{global_step},,{mean_vloss}\n")

                    # checkpoint at validation time
                    save_checkpoint(rank, model, optimizer, epoch, global_step, args.output_dir)

                    model.train()

                # sync again so all ranks leave validation at the same time
                if world_size > 1:
                    dist.barrier()

        # final epoch checkpoint
        if rank == 0:
            save_checkpoint(rank, model, optimizer, epoch, global_step, args.output_dir)

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
# src/train_mlm.py

import os
import argparse
import lightning as L
from transformers import AutoTokenizer

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from model import MLMLitModule
from dataset import MLMDataModule

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.set_float32_matmul_precision("high")
# torch.set_float32_matmul_precision("medium")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_parquet", type=str, required=True)
    parser.add_argument("--valid_parquet", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="aaronfeller/PeptideMTR_sm")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_nodes", type=int, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id

    # Lightning DataModule
    dm = MLMDataModule(
        args.train_parquet,
        args.valid_parquet,
        tokenizer,
        args.batch_size,
        mask_token_id,
        pad_token_id,
    )

    # LightningModel
    lit_model = MLMLitModule(tokenizer, lr=args.lr)

    # Callbacks
    ckpt = ModelCheckpoint(
        dirpath=args.output_dir,
        save_last=True,
        save_top_k=5,
        monitor="valid_loss",
        mode="min",
        filename="epoch-{epoch:03d}-vloss-{valid_loss:.4f}",
    )

    logger = CSVLogger(args.output_dir, name="logs")

    # Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",
        num_nodes=args.num_nodes,
        strategy="ddp",       # automatically uses RANK + WORLD_SIZE from torchrun
        max_epochs=args.epochs,
        val_check_interval=0.1,
        limit_val_batches=0.2,
        accumulate_grad_batches=1,
        precision="bf16-mixed",   # optional
        callbacks=[ckpt],
        logger=logger,
    )

    trainer.fit(lit_model, dm)


if __name__ == "__main__":
    main()

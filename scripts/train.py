# Timing
import datetime
print("Current Time =", datetime.datetime.now().strftime("%H:%M:%S"))
print("Loading libraries...")

# Import libraries
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.pytorch.profilers import SimpleProfiler
from torchvision.transforms import Compose
from transformers import AutoTokenizer, DataCollatorWithPadding #, RobertaForMaskedLM, RobertaConfig

# Import local functions and classes
from model.model import model_config, MLM_model


# Define the main function
def main():
    
    # Load Tokenizer ####################################################################
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        "aaronfeller/PeptideMTR_sm",
        padding=False,                # don’t pad now
        truncation=True,              # allow truncation
        max_length=2048               # optional cap
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)
    
    # Data loader ###############################################################
    csv_file = '../data/'
    composed = Compose([SMILES_to_input(tokenizer)])
    
    dataset = SMILES_Dataset(csv_file, transform=composed)
    
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=8, batch_size=64, shuffle=False)
    
    # Model parameters ###############################################################
    
    config = model_config    
    model = MLM_model(config=config)
    
    # initialize to best practices
    def initialize_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    initialize_parameters(model)
    

    # Training ###############################################################
    print("Training model...")
    print("Current Time =", datetime.datetime.now().strftime("%H:%M:%S"))
    
    torch.set_float32_matmul_precision('medium') # medium, high, or highest available (medium for speed, highest for accuracy)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{args.directory}',
        filename='pepLM-{epoch:02d}-{step:.0f}-{val_loss:.3f}',
        save_top_k=30,
        mode='min'
    )
        
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    profiler = SimpleProfiler(dirpath=f'checkpoints/{args.directory}', filename='prof_logs')
    
    trainer = pl.Trainer(
        max_epochs=3,
        log_every_n_steps=10,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        val_check_interval=0.1,
        # check_val_every_n_epoch=1,
        accelerator='gpu',
        devices=-1, # for all GPUs
        enable_checkpointing=True,
        default_root_dir="checkpoints/",
        callbacks=[checkpoint_callback, lr_monitor],
        precision='16-mixed',
        profiler= profiler
        )
    
    trainer.fit(
        model=pepLM_model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        )
    
    print("Current Time =", datetime.datetime.now().strftime("%H:%M:%S"))
    print("Done!")

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
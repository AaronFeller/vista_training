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
from lightning.pytorch.profilers import SimpleProfiler
from torchvision.transforms import Compose
from transformers import AutoTokenizer#, RobertaForMaskedLM, RobertaConfig
from roformer import RoFormerForMaskedLM, RoFormerConfig

# Import local functions and classes
from functions.data_funcs import * # import functions.data_funcs as dfunc
from models.my_model import *
from dataloader.dataloader import *

torch.cuda.empty_cache()

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# print('Cuda available: ', torch.cuda.is_available())
# print('PyTorch version is: ', torch.__version__)

# Define the main function
def main():
    # set up arguments ##############################################################
    parser = argparse.ArgumentParser(description='Train a Transformer model on a peptide dataset')
    parser.add_argument('-d', '--directory', type=str, required=True, 
                        help='Directory name.')
    args = parser.parse_args()
    
    # Load Tokenizer ####################################################################
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained("aaronfeller/PeptideMTR_tokenizerOnly",
                                              pad_sequence_length=max_length, padding_side='right', 
                                              truncation=True, max_length=2048, 
                                              return_tensors='pt')

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
    
    config = RoFormerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=514,
        num_hidden_layers=6,
        num_attention_heads=12,
        hidden_size=768,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
        is_decoder=False,
    )
    
    model = RoFormerForMaskedLM(config=config)
    model = reset_parameters(model)
    
    # Count model parameters
    print(f"Number of parameters: {count_parameters(model)/1e6:.2f}M")
    
    pepLM_model = pepLM(model) # , vocab_size)
    
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
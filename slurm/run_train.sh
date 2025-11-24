#!/bin/bash
#SBATCH -J mlm_single
#SBATCH -o mlm_single.o%j
#SBATCH -e mlm_single.e%j
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 6-12:00:00
#SBATCH -A MCB24088
#SBATCH --mail-type=END,FAIL        # Notifications
#SBATCH --mail-user=aaron.feller@utexas.edu

module purge
module load gcc/15.1.0
module load cuda
module load python3/3.11.8
source $SCRATCH/envs/protein-mlm/bin/activate

cd $WORK/SMILES_ESM/protein_mlm

# how many GPUs on the node? e.g. 4
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

torchrun \
  --nproc_per_node=$NUM_GPUS \
  --nnodes=1 \
  --node_rank=0 \
  src/train_mlm.py \
  --train_csv data/train.csv \
  --output_dir checkpoints/mlm_single_node
#!/bin/bash
#SBATCH -J 2gpu_test2
#SBATCH -A MCB24088
#SBATCH -p gh-dev
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aaron.feller@utexas.edu

set -e

cd $SLURM_SUBMIT_DIR
mkdir -p logs

export PYTHONPATH=$SLURM_SUBMIT_DIR/src:$PYTHONPATH

module load gcc cuda python3
source /scratch/08929/afeller/envs/protein-mlm/bin/activate

srun python3 src/train_mlm.py \
    --train_parquet data/train/train_smiles.parquet \
    --valid_parquet data/test/test_smiles.parquet \
    --tokenizer_name aaronfeller/PeptideMTR_sm \
    --batch_size 32 \
    --epochs 5 \
    --lr 1e-4 \
    --output_dir ${SLURM_JOB_NAME}_output \
    --num_nodes $SLURM_JOB_NUM_NODES

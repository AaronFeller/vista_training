#!/bin/bash
#SBATCH -J 2gpu_test
#SBATCH -A MCB24088
#SBATCH -p gh-dev
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aaron.feller@utexas.edu

set -e

cd $SLURM_SUBMIT_DIR
mkdir -p logs

export PYTHONPATH=$SLURM_SUBMIT_DIR/src:$PYTHONPATH

echo "Loading modules..."
module load gcc cuda python3

echo "Activating venv..."
source /scratch/08929/afeller/envs/protein-mlm/bin/activate

# Master is first node
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE=$SLURM_NTASKS

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"

echo "Launching distributed job via srun..."

srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=1 \
     --cpu-bind=none \
     bash -c '
        export RANK=$SLURM_PROCID
        export LOCAL_RANK=0

        torchrun \
            --nnodes='"$SLURM_JOB_NUM_NODES"' \
            --nproc_per_node=1 \
            --node_rank=$SLURM_PROCID \
            --master_addr='"$MASTER_ADDR"' \
            --master_port='"$MASTER_PORT"' \
            src/train_mlm.py \
                --train_parquet data/train/train_smiles.parquet \
                --valid_parquet data/test/test_smiles.parquet \
                --tokenizer_name aaronfeller/PeptideMTR_sm \
                --batch_size 32 \
                --epochs 5 \
                --lr 1e-4 \
                --output_dir test_out_2gpu \
     '
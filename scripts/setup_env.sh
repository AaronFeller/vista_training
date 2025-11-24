# 0.2: Load modules
module purge
module load gcc/15.1.0
module load cuda
module load python3/3.11.8

# 0.3: Create venv in scratch
mkdir -p $SCRATCH/envs
python3 -m venv $SCRATCH/envs/protein-mlm

# 0.4: Activate and install packages
source $SCRATCH/envs/protein-mlm/bin/activate

pip install --upgrade pip
# GPU torch (CUDA 12.9 wheel index per Vista docs)
pip install "torch" "torchvision" --index-url https://download.pytorch.org/whl/cu129

# Core stack
pip install transformers datasets accelerate einops pandas numpy scikit-learn tqdm
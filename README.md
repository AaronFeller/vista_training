# smiles-esm — Vista training scaffold

This repo is set up to train PyTorch/Lightning models on TACC Vista (GH/H200 nodes) using `uv` for environment management.

## Vista setup (one-time)

On Vista:

```bash
# 1. Clone the repo into $WORK or $SCRATCH
cd $SCRATCH        # or $WORK
git clone git@github.com:YOURNAME/smiles-esm.git
cd smiles-esm

# 2. Start an interactive dev session on a compute node
idev -p gh-dev -N 1 -n 1 -t 01:00:00

# 3. Inside the compute node, set up env
bash scripts/setup_vista_env.sh

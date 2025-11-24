#!/usr/bin/env bash
set -e

#=== CONFIGURATION ===#
IMAGE_NAME="vista_env"            # directory name for sandbox container
BASE_IMAGE="docker://nvcr.io/nvidia/pytorch:24.12-py3"   # adjust if you prefer a different base
PYTHON_PKGS="numpy scipy pandas transformers torch torchvision"  # list your frequently used packages
ENV_DIR="/work/08929/afeller/containers"            # directory where you want to store your container
#=====================#

mkdir -p "${ENV_DIR}"
cd "${ENV_DIR}"

echo "✔ Building sandbox container: ${IMAGE_NAME}"
apptainer build --sandbox "${IMAGE_NAME}" "${BASE_IMAGE}"

echo "✔ Entering the sandbox container to install packages"
apptainer shell --writable --nv "${IMAGE_NAME}" /bin/bash <<'EOF'
# Inside container
python3 -m pip install --upgrade pip
python3 -m pip install ${PYTHON_PKGS}
# you can add any additional setup, e.g., installing JupyterLab
# python3 -m pip install jupyterlab
EOF

echo "✔ Build complete. To use this container, run:"
echo "    apptainer shell --nv ${ENV_DIR}/${IMAGE_NAME}"
echo "  or for batch jobs:"
echo "    apptainer exec --nv ${ENV_DIR}/${IMAGE_NAME} <your-command>"

echo "✔ Tip: If you want to make an immutable SIF image for production use, you can do:"
echo "    apptainer build vista_env.sif ${ENV_DIR}/${IMAGE_NAME}"
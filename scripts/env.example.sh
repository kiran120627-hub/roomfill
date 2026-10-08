# Copy to ~/hn3d/env.sh and `source` it before running the pipeline (WSL Ubuntu).
# CUDA 13.1 nvcc from apt + the CUDA 13 headers/libs that ship inside the PyTorch cu130 wheels.
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$HOME/hn3d/bin:$PATH
SP=$HOME/hn3d/lib/python3.12/site-packages
export CPATH=$SP/nvidia/cu13/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$SP/nvidia/cu13/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export TORCH_CUDA_ARCH_LIST=12.0          # RTX 50-series (Blackwell); use your GPU's compute capability
export MAX_JOBS=8
export LAMA_MODEL=$HOME/models/big-lama.pt
# gsplat build (from source, only the kernels this project uses):
#   sed -i 's/define GSPLAT_NUM_CHANNELS .*/define GSPLAT_NUM_CHANNELS 1, 3, 4/' gsplat/cuda/csrc/Config.h
#   BUILD_3DGUT=0 BUILD_2DGS=0 BUILD_EXPERIMENTAL=0 pip install --no-build-isolation ./gsplat   (needs ninja)

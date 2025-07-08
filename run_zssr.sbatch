#!/bin/bash
#SBATCH --job-name=zssr_tensorflow
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=02:00:00

echo "============================================"
echo "ZSSR TensorFlow Job Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Memory allocated: 32GB per GPU"
echo "============================================"

# Проверим среду в TensorFlow контейнере
echo "Testing environment in TensorFlow container:"
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import sys
print(f'Python: {sys.version}')

try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except Exception as e:
    print(f'❌ NumPy: {e}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU count: {torch.cuda.device_count()}')
        print(f'✅ GPU name: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch: {e}')

try:
    import cv2
    print(f'✅ OpenCV: {cv2.__version__}')
except Exception as e:
    print(f'❌ OpenCV: {e}')
"

echo "============================================"
echo "Starting ZSSR training with TensorFlow container:"

# Запускаем ZSSR в TensorFlow контейнере
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $HOME/ZRRS_Attention:$HOME/ZRRS_Attention \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python $HOME/ZRRS_Attention/zssr_attention_dt.py

echo "============================================"
echo "ZSSR TensorFlow Job Finished: $(date)"
echo "============================================"

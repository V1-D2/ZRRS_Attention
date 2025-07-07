#!/bin/bash
#SBATCH --job-name=zssr_attention
#SBATCH --partition=salvador        
#SBATCH --gres=gpu:a100:1           
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                  
#SBATCH --time=02:00:00             
#SBATCH --output=logs/%x-%j.out

echo "GPUs assigned to this job: $CUDA_VISIBLE_DEVICES"

singularity exec --nv \
  --bind $HOME:/home/vdidur \
  $HOME/containers/pytorch-cu12.sif \
  python /home/vdidur/ZRRS_Attention/zssr_attention.py

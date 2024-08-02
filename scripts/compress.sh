#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00


# Activate Anaconda work environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate c4c


python preprocess/compress_video.py --input_root /home/it21902/V2T-Action-Graph-JKSUCIS-2023/dataset/MSVD/raw --output_root compressed_videos

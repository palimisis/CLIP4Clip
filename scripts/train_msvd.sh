#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00             
      
# Activate Anaconda work environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate c4c

#python -m torch.distributed.launch
torchrun --nproc_per_node=1 \
    main_task_retrieval.py --do_train --num_thread_reader=4 \
    --epochs=5 --batch_size=16 --n_display=50 \
    --data_path "/home/it21902/CLIP4Clip/msvd_data" \
    --features_path "/home/it21902/CLIP4Clip/compressed_videos" \
    --output_dir ckpts/ckpt_msvd_retrieval_looseType \
    --lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
    --datatype msvd \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header meanP \
    --pretrained_clip_name "ViT-B/16"

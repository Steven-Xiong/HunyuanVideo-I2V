
# BSUB -o ./bjob_logs/train_physion_4.3.%J

# BSUB -q gpu-compute


# BSUB -gpu "num=1:mode=shared:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_80GB" 
# BSUB -J hunyuan


source ~/.bashrc
conda activate diffusion-pipe
cd /project/osprey/scratch/x.zhexiao/video_gen/HunyuanVideo-I2V


# Set environment variables
# export HOST_GPU_NUM=2  # Set the number of GPUs to use

# # Run extraction script
# bash hyvideo/hyvae_extract/start.sh

#### single gpu extraction
export PYTHONPATH=${PYTHONPATH}:`pwd`
export HOST_GPU_NUM=1
CUDA_VISIBLE_DEVICES=0 python3 -u hyvideo/hyvae_extract/run.py --local_rank 0 --config 'hyvideo/hyvae_extract/vae.yaml'

# sh scripts/run_train_image2video_lora.sh
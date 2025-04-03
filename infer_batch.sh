# BSUB -o ./bjob_logs/train_physion_3.25.%J

# BSUB -q gpu-compute

# BSUB -gpu "num=4:mode=shared:j_exclusive=yes:gmodel=NVIDIAA10080GBPCIe" 
# BSUB -J hunyuan


source ~/.bashrc
conda activate HunyuanVideo
cd /project/osprey/scratch/x.zhexiao/video_gen/HunyuanVideo-I2V

# python3 infer_batch.py \
#     --model HYVideo-T/2 \
#     --input_folder example \
#     --caption_prefix " " \
#     --i2v-mode \
#     --i2v-resolution 720p \
#     --i2v-stability \
#     --infer-steps 50 \
#     --video-length 129 \
#     --flow-reverse \
#     --flow-shift 7.0 \
#     --seed 0 \
#     --embedded-cfg-scale 6.0 \
#     --use-cpu-offload \
#     --save-path ./results/

torchrun --nproc_per_node=4 infer_batch.py \
    --model HYVideo-T/2 \
    --input_folder example \
    --caption_prefix " " \
    --i2v-mode \
    --i2v-resolution 720p \
    --i2v-stability \
    --infer-steps 50 \
    --video-length 129 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --save-path ./results \
    --ulysses-degree 4 \
    --ring-degree 1 \
    --video-size 1280 720 \
    --xdit-adaptive-size
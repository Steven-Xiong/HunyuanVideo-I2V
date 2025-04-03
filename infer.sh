# BSUB -o ./bjob_logs/train_physion_3.25.%J

# BSUB -q gpu-compute

# BSUB -gpu "num=1:mode=shared:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_80GB" 
# BSUB -J hunyuan


source ~/.bashrc
conda activate HunyuanVideo
cd /project/osprey/scratch/x.zhexiao/video_gen/HunyuanVideo-I2V


# cd HunyuanVideo-I2V

python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --prompt "An Asian man with short hair in black tactical uniform and white clothes waves a firework stick." \
    --i2v-mode \
    --i2v-image-path ./assets/demo/i2v/imgs/0.jpg \
    --i2v-resolution 720p \
    --i2v-stability \
    --infer-steps 50 \
    --video-length 129 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --use-cpu-offload \
    --save-path ./results


##### high-dynamic ############
# cd HunyuanVideo-I2V

# python3 sample_image2video.py \
#     --model HYVideo-T/2 \
#     --prompt "An Asian man with short hair in black tactical uniform and white clothes waves a firework stick." \
#     --i2v-mode \
#     --i2v-image-path ./assets/demo/i2v/imgs/0.jpg \
#     --i2v-resolution 720p \
#     --infer-steps 50 \
#     --video-length 129 \
#     --flow-reverse \
#     --flow-shift 17.0 \
#     --embedded-cfg-scale 6.0 \
#     --seed 0 \
#     --use-cpu-offload \
#     --save-path ./results

##### multi-gpu ###########

# cd HunyuanVideo-I2V

# torchrun --nproc_per_node=8 sample_image2video.py \
#     --model HYVideo-T/2 \
#     --prompt "An Asian man with short hair in black tactical uniform and white clothes waves a firework stick." \
#     --i2v-mode \
#     --i2v-image-path ./assets/demo/i2v/imgs/0.jpg \
#     --i2v-resolution 720p \
#     --i2v-stability \
#     --infer-steps 50 \
#     --video-length 129 \
#     --flow-reverse \
#     --flow-shift 7.0 \
#     --seed 0 \
#     --embedded-cfg-scale 6.0 \
#     --save-path ./results \
#     --ulysses-degree 8 \
#     --ring-degree 1 \
#     --video-size 1280 720 \
#     --xdit-adaptive-size
#!/usr/bin/env python3
import os
import time
import base64
from pathlib import Path
from loguru import logger
from datetime import datetime
import argparse

from hyvideo.utils.file_utils import save_videos_grid
# 从 hyvideo.config 导入所有必要的参数添加函数与 sanity_check
from hyvideo.config import (
    add_network_args,
    add_extra_models_args,
    add_denoise_schedule_args,
    add_i2v_args,
    add_lora_args,
    add_inference_args,
    add_parallel_args,
    sanity_check_args,
)
from hyvideo.inference import HunyuanVideoSampler

def encode_video_to_data_url(video_path):
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    encoded = base64.b64encode(video_bytes).decode("utf-8")
    return f"data:video/mp4;base64,{encoded}"

def parse_args():
    parser = argparse.ArgumentParser(
        description="HunyuanVideo inference/lora training script with optional batch mode."
    )
    # 添加原有参数
    parser = add_network_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_i2v_args(parser)
    parser = add_lora_args(parser)
    parser = add_inference_args(parser)
    parser = add_parallel_args(parser)
    # 增加批量处理相关参数：仅新增 --input_folder 和 --caption_prefix
    parser.add_argument(
        "--input_folder",
        type=str,
        default="",
        help="Folder containing images and corresponding txt files for batch mode."
    )
    parser.add_argument(
        "--caption_prefix",
        type=str,
        default="",
        help="Optional prefix to add to each caption in batch mode."
    )
    args = parser.parse_args()
    args = sanity_check_args(args)
    return args

def main():
    args = parse_args()
    logger.info(args)
    
    # 多卡时仅由 rank0 保存输出，其他进程直接退出
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank != 0:
        logger.info(f"Non-zero rank ({local_rank}), skipping saving outputs.")
        return

    # 判断是否批量模式：当 --input_folder 不为空时启用批量模式
    batch_mode = args.input_folder.strip() != ""
    
    # 合并输出目录：统一使用 args.save_path 作为输出目录
    output_dir = args.save_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output videos and HTML will be saved to: {output_dir}")
    
    # 模型根目录使用 --model-base 参数（请确保该路径存在）
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"Model path does not exist: {models_root_path}")
    
    # 加载模型
    sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    # 模型加载后可能更新了一部分参数
    args = sampler.args

    if batch_mode:
        html_entries = []
        # 查找 input_folder 中所有图片（支持 jpg/jpeg/png/bmp）
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [str(p) for p in Path(args.input_folder).glob('*') if p.suffix.lower() in image_extensions]
        image_files.sort()
        logger.info(f"Found {len(image_files)} images in {args.input_folder}")
        
        for img_path in image_files:
            # 假设每个图片有同名的 txt 文件存放 caption
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if not os.path.exists(txt_path):
                logger.warning(f"No caption file found for image {img_path}, skipping.")
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            # 如果有 caption_prefix，则拼接在前面
            full_prompt = (args.caption_prefix if args.caption_prefix.strip() != "" else "") + caption
            logger.info(f"Processing image {img_path} with prompt: {full_prompt}")
            
            # 更新当前样本参数：prompt 与 i2v_image_path
            args.prompt = full_prompt
            args.i2v_image_path = img_path
            
            # 调用 predict，参数与原始代码保持一致
            outputs = sampler.predict(
                prompt=args.prompt, 
                height=args.video_size[0],
                width=args.video_size[1],
                video_length=args.video_length,
                seed=args.seed,
                negative_prompt=args.neg_prompt,
                infer_steps=args.infer_steps,
                guidance_scale=args.cfg_scale,
                num_videos_per_prompt=args.num_videos,
                flow_shift=args.flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=args.embedded_cfg_scale,
                i2v_mode=args.i2v_mode,
                i2v_resolution=args.i2v_resolution,
                i2v_image_path=args.i2v_image_path,
                i2v_condition_type=args.i2v_condition_type,
                i2v_stability=args.i2v_stability,
                ulysses_degree=args.ulysses_degree,
                ring_degree=args.ring_degree,
                xdit_adaptive_size=args.xdit_adaptive_size
            )
            # 按原始逻辑，取生成的第一个视频
            sample = outputs['samples'][0].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            sanitized_prompt = full_prompt.replace("/", "").replace(" ", "_")[:100]
            cur_save_path = os.path.join(output_dir, f"{time_flag}_seed{outputs['seeds'][0]}_{sanitized_prompt}.mp4")
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f"Saved generated video to: {cur_save_path}")
            
            # 采用编码方式，将生成视频转换为 data URL
            gen_video_data_url = encode_video_to_data_url(cur_save_path)
            
            # 尝试查找 ground truth 视频（假设 ground truth 视频路径与图片同名、扩展名为 .mp4）
            gt_video_path = os.path.splitext(img_path)[0] + ".mp4"
            if os.path.exists(gt_video_path):
                gt_video_data_url = encode_video_to_data_url(gt_video_path)
            else:
                gt_video_data_url = ""
            
            # 构造 HTML 条目：同时展示 prompt、生成视频和 ground truth 视频
            entry_html = f'''
            <div style="display: flex; align-items: flex-start; margin-bottom: 20px;">
                <div style="flex: 1; margin-right: 20px;">
                    <h3>Caption</h3>
                    <p>{full_prompt}</p>
                </div>
                <div style="flex: 1; margin-right: 20px;">
                    <h3>Generated Video</h3>
                    <video width="320" height="240" controls>
                        <source src="{gen_video_data_url}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
                <div style="flex: 1;">
                    <h3>Ground Truth Video</h3>
                    <video width="320" height="240" controls>
                        <source src="{gt_video_data_url}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
            </div>
            '''
            html_entries.append(entry_html)
        
        html_content = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Batch Video Generation Results</title>
        </head>
        <body>
            <h1>Batch Video Generation Results</h1>
            {"".join(html_entries)}
        </body>
        </html>
        '''
        html_save_path = os.path.join(output_dir, "batch_results.html")
        with open(html_save_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"HTML results saved to: {html_save_path}")
    else:
        outputs = sampler.predict(
            prompt=args.prompt,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
            i2v_mode=args.i2v_mode,
            i2v_resolution=args.i2v_resolution,
            i2v_image_path=args.i2v_image_path,
            i2v_condition_type=args.i2v_condition_type,
            i2v_stability=args.i2v_stability,
            ulysses_degree=args.ulysses_degree,
            ring_degree=args.ring_degree,
            xdit_adaptive_size=args.xdit_adaptive_size
        )
        samples = outputs['samples']
        for i, sample in enumerate(samples):
            sample = sample.unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            cur_save_path = f"{args.save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f"Saved sample video to: {cur_save_path}")

if __name__ == "__main__":
    main()

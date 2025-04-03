import json
import os

# 指定输入和输出文件的路径
input_file_path = "data/physion/captions_v3_78B_short.json"   # 替换为你的输入 JSON 文件路径
output_dir = "data/physion/hunyuan"  # 替换为你希望保存的输出文件路径


# 要添加的路径前缀
prefix = "data/physion/Physion/"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取原始 JSON 文件
with open(input_file_path, "r", encoding="utf-8") as infile:
    data = json.load(infile)

# 指定 meta_file.list 文件路径
meta_file_path = os.path.join(output_dir, "meta_file.list")

# 使用写模式打开 meta_file.list 文件（如果文件不存在，会自动创建）
with open(meta_file_path, "w", encoding="utf-8") as meta_file:
    # 遍历每个条目
    for entry in data:
        original_video_path = entry.get("video_path", "")
        # 在原有 video_path 前添加指定前缀
        full_video_path = prefix + original_video_path
        
        # 去除 caption 字符串两端可能存在的双引号
        caption = entry.get("caption", "").strip('"')
        
        # 构造目标格式数据
        output_item = {
            "video_path": full_video_path,
            "raw_caption": {
                "long caption": caption
            }
        }
        
        # 根据 original_video_path 的文件名生成对应的 JSON 文件名
        # 例如 "Contain/mp4s/pilot-containment-multi-bowl_0012_img.mp4" 转换为 "pilot-containment-multi-bowl_0012_img.json"
        base_name = os.path.basename(original_video_path)
        file_name_no_ext, _ = os.path.splitext(base_name)
        json_file_name = f"{file_name_no_ext}.json"
        json_file_path = os.path.join(output_dir, json_file_name)
        
        # 将转换后的数据写入单独的 JSON 文件中
        with open(json_file_path, "w", encoding="utf-8") as outfile:
            json.dump(output_item, outfile, ensure_ascii=False, indent=2)
        
        # 将生成的 JSON 文件路径写入 meta_file.list，每行一个文件路径
        meta_file.write(json_file_path + "\n")

print("转换完成！")
print("生成的 JSON 文件存储在：", output_dir)
print("meta_file.list 文件已创建，路径为：", meta_file_path)

# # 读取原始 JSON 文件
# with open(input_file_path, "r", encoding="utf-8") as infile:
#     input_data = json.load(infile)

# # 转换数据格式
# output_data = []
# for item in input_data:
#     # 移除 caption 字符串开头和结尾的引号（如果存在）
#     caption_text = item.get("caption", "").strip('"')
#     new_item = {
#         "video_path": item.get("video_path", ""),
#         "raw_caption": {
#             "long caption": caption_text
#         }
#     }
#     output_data.append(new_item)

# # 将转换后的数据写入到指定输出文件
# with open(output_file_path, "w", encoding="utf-8") as outfile:
#     json.dump(output_data, outfile, ensure_ascii=False, indent=2)

# print("转换成功，已保存至：", output_file_path)

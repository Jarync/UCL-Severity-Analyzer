import os
import pandas as pd
import numpy as np
from PIL import Image
import uuid
from datetime import datetime
from tqdm import tqdm
import shutil
import hashlib  # 添加hashlib用于计算图像哈希值

# 添加计算图像哈希值的函数
def calculate_image_hash(img_path):
    """
    计算图像的哈希值，用于识别重复图像
    
    参数:
    - img_path: 图像文件路径
    
    返回:
    - hash_value: 图像的MD5哈希值
    """
    try:
        with Image.open(img_path) as img:
            # 转换为RGB并缩小尺寸，减少计算量但保留特征
            img = img.convert('RGB').resize((64, 64), Image.LANCZOS)
            img_data = np.array(img)
            # 使用MD5计算哈希值
            return hashlib.md5(img_data.tobytes()).hexdigest()
    except Exception as e:
        print(f"计算图像哈希值时出错 {img_path}: {str(e)}")
        return None

def process_images(input_folder, output_folder, metadata_file, target_size=512, force_reprocess=False):
    """
    预处理图片并保存元数据
    
    参数:
    - input_folder: 原始图片文件夹路径
    - output_folder: 处理后图片输出文件夹路径
    - metadata_file: 元数据CSV文件路径
    - target_size: 图片处理后的目标尺寸 (默认512×512)
    - force_reprocess: 是否强制重新处理所有图片(默认False)
    """
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")
    
    # 如果强制重新处理，清空输出文件夹和元数据
    if force_reprocess:
        # 删除输出文件夹中的所有文件
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"删除旧文件: {file_path}")
            except Exception as e:
                print(f"删除文件 {file_path} 时出错: {str(e)}")
        
        # 清空或创建新的元数据文件
        metadata_df = pd.DataFrame(columns=[
            "图像ID", "原始文件名", "原始宽度", "原始高度",
            "处理后宽度", "处理后高度", "缩放比例X", "缩放比例Y",
            "中心偏移X", "中心偏移Y", "处理时间", "图像哈希值"
        ])
        metadata_df.to_csv(metadata_file, index=False, encoding='utf-8-sig')
        print(f"已创建新的元数据文件: {metadata_file}")
    else:
        # 创建或加载元数据文件
        if os.path.exists(metadata_file):
            metadata_df = pd.read_csv(metadata_file)
            print(f"加载现有元数据文件，当前有 {len(metadata_df)} 条记录")
            
            # 如果旧元数据没有图像哈希值列，添加该列
            if "图像哈希值" not in metadata_df.columns:
                metadata_df["图像哈希值"] = ""
                print("元数据文件中添加了'图像哈希值'列")
        else:
            metadata_df = pd.DataFrame(columns=[
                "图像ID", "原始文件名", "原始宽度", "原始高度",
                "处理后宽度", "处理后高度", "缩放比例X", "缩放比例Y",
                "中心偏移X", "中心偏移Y", "处理时间", "图像哈希值"
            ])
            print("创建新的元数据文件")
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"找到 {len(image_files)} 张图片需要处理")
    
    # 记录新处理的图片
    new_records = []
    processed_count = 0
    skipped_count = 0
    duplicate_count = 0
    
    # 计算现有处理过图片的哈希值
    existing_hashes = set()
    if not force_reprocess and "图像哈希值" in metadata_df.columns:
        existing_hashes = set(metadata_df["图像哈希值"].dropna())
        
        # 如果元数据中有空的哈希值，计算并填充
        if metadata_df["图像哈希值"].isna().any():
            print("更新现有图片的哈希值...")
            for idx, row in metadata_df.iterrows():
                if pd.isna(row["图像哈希值"]):
                    # 查找处理后的图片文件
                    processed_files = [f for f in os.listdir(output_folder) 
                                      if f.startswith(row["图像ID"])]
                    if processed_files:
                        img_path = os.path.join(output_folder, processed_files[0])
                        hash_val = calculate_image_hash(img_path)
                        metadata_df.at[idx, "图像哈希值"] = hash_val
                        if hash_val:
                            existing_hashes.add(hash_val)
            
            # 保存更新后的元数据
            metadata_df.to_csv(metadata_file, index=False, encoding='utf-8-sig')
            print(f"已更新现有图片的哈希值，共有 {len(existing_hashes)} 个唯一哈希值")
    
    for img_file in tqdm(image_files, desc="处理图片"):
        img_path = os.path.join(input_folder, img_file)
        
        try:
            # 先尝试打开图片，确保可以访问
            with Image.open(img_path) as _:
                pass
                
            # 计算当前图片的哈希值
            img_hash = calculate_image_hash(img_path)
            
            # 如果不是强制重新处理，检查是否已处理过（按文件名或哈希值）
            if not force_reprocess:
                if img_file in metadata_df['原始文件名'].values:
                    print(f"跳过已处理图片(文件名匹配): {img_file}")
                    skipped_count += 1
                    continue
                
                if img_hash in existing_hashes:
                    print(f"跳过已处理图片(内容重复): {img_file}")
                    duplicate_count += 1
                    continue
            
            # 生成唯一ID
            image_id = str(uuid.uuid4())[:8]
            
            # 使用PIL读取图片
            pil_img = Image.open(img_path)
            
            # 转换为RGB模式
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # 获取原始尺寸
            orig_width, orig_height = pil_img.size
            
            # 计算缩放比例，保持长宽比
            scale_x = target_size / orig_width
            scale_y = target_size / orig_height
            scale = min(scale_x, scale_y)
            
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # 缩放图片，使用高质量插值
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # 创建白色背景的新图片
            new_img = Image.new('RGB', (target_size, target_size), (255, 255, 255))
            
            # 将缩放后的图片居中放置
            paste_x = (target_size - new_width) // 2
            paste_y = (target_size - new_height) // 2
            
            new_img.paste(pil_img, (paste_x, paste_y))
            
            # 保存处理后的图片
            output_filename = f"{image_id}_{img_file}"
            output_path = os.path.join(output_folder, output_filename)
            new_img.save(output_path, quality=95)  # 保持高质量
            
            # 记录元数据
            new_record = {
                "图像ID": image_id,
                "原始文件名": img_file,
                "原始宽度": orig_width,
                "原始高度": orig_height,
                "处理后宽度": target_size,
                "处理后高度": target_size,
                "缩放比例X": scale,
                "缩放比例Y": scale,
                "中心偏移X": paste_x,
                "中心偏移Y": paste_y,
                "处理时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "图像哈希值": img_hash
            }
            
            new_records.append(new_record)
            processed_count += 1
            
            # 添加新哈希值到已存在集合中
            if img_hash:
                existing_hashes.add(img_hash)
            
        except Exception as e:
            print(f"处理图片 {img_file} 时出错: {str(e)}")
    
    # 更新元数据
    if new_records:
        new_df = pd.DataFrame(new_records)
        if force_reprocess:
            metadata_df = new_df
        else:
            metadata_df = pd.concat([metadata_df, new_df], ignore_index=True)
        metadata_df.to_csv(metadata_file, index=False, encoding='utf-8-sig')
    
    print(f"处理完成！成功处理 {processed_count} 张新图片")
    if not force_reprocess:
        print(f"跳过 {skipped_count} 张已处理图片(文件名匹配)")
        print(f"跳过 {duplicate_count} 张重复内容图片(哈希值匹配)")
    print(f"元数据已保存至: {metadata_file}")
    
    return processed_count, skipped_count + duplicate_count

# 主函数
if __name__ == "__main__":
    print("=" * 50)
    print("唇裂项目图片处理工具")
    print("=" * 50)
    
    # 直接使用绝对路径
    input_folder = r"D:\upm_model train\Second_model\原始图片"
    output_folder = r"D:\upm_model train\Second_model\处理后图片"
    metadata_file = r"D:\upm_model train\Second_model\图像元数据.csv"
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误：原始图片文件夹不存在: {input_folder}")
        exit(1)
    
    # 是否强制重新处理所有图片
    force_reprocess = False
    user_input = input("是否强制重新处理所有图片? (y/n, 默认n): ").strip().lower()
    if user_input == 'y':
        force_reprocess = True
        print("已启用强制重新处理，将处理所有图片")
    else:
        print("增量处理模式：只处理新添加的图片，已处理的图片将被跳过")
    
    print(f"开始处理 {input_folder} 中的图片...")
    print(f"目标尺寸: 512 × 512 像素")
    
    # 处理图片
    processed, skipped = process_images(
        input_folder, 
        output_folder,
        metadata_file,
        target_size=512, 
        force_reprocess=force_reprocess
    )
    
    # 处理完成后统计结果
    print(f"处理完成！新增 {processed} 张图片，跳过 {skipped} 张图片")
    
    # 读取元数据统计
    try:
        if os.path.exists(metadata_file):
            metadata = pd.read_csv(metadata_file)
            print(f"处理结果统计: 共 {len(metadata)} 张")
    except Exception as e:
        print(f"读取元数据统计失败: {str(e)}")
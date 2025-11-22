import json
import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_second_model_json(json_file_path, output_csv_path, image_folder=None):
    """处理第二个模型的Label Studio导出的JSON文件，生成关键点坐标CSV"""
    
    # 检查文件路径是否存在
    if not os.path.exists(json_file_path):
        print(f"错误：找不到JSON文件: {json_file_path}")
        return None
    
    # 检查图像文件夹是否存在
    if image_folder and not os.path.exists(image_folder):
        print(f"警告：找不到图像文件夹: {image_folder}")
    
    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"读取到 {len(data)} 张图片的标注数据")
    except Exception as e:
        print(f"读取JSON文件出错: {str(e)}")
        return None
    
    # 准备结果数据
    results = []
    
    for item in data:
        # 处理每张图片
        original_filename = item['file_upload']
        image_id = original_filename.split('_')[0]
        
        print(f"处理图片: {original_filename} (ID: {image_id})")
        
        if 'annotations' in item and len(item['annotations']) > 0:
            keypoints = {}
            
            # 获取所有关键点
            for point in item['annotations'][0]['result']:
                if point['type'] == 'keypointlabels':
                    label = point['value']['keypointlabels'][0]
                    x = point['value']['x'] / 100  # 归一化到0-1
                    y = point['value']['y'] / 100
                    keypoints[label] = (x, y)
            
            # 添加结果
            for label, (x, y) in keypoints.items():
                results.append({
                    '图像ID': image_id,
                    '原始文件名': original_filename,
                    '关键点': label,
                    '归一化X': x,
                    '归一化Y': y,
                    '像素X': round(x * 512),  # 假设图像大小为512x512
                    '像素Y': round(y * 512)
                })
        else:
            print(f"  警告: 图片 {original_filename} 缺少必要的关键点")
    
    # 转换为DataFrame并保存CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"标注数据已保存至: {output_csv_path}")
        
        # 显示统计信息
        image_count = len(set(df['图像ID']))
        
        print(f"\n统计信息:")
        print(f"总标注图片数: {image_count}")
        
        return df
    else:
        print("警告: 没有处理任何有效数据")
        return None

# 主程序执行
if __name__ == "__main__":
    # 设置文件路径
    json_file = r"D:\upm_model train\Second_model\project-4-at-2025-06-13-01-11-841a8364.json"
    project_dir = r"D:\upm_model train\Second_model"
    output_csv = os.path.join(project_dir, "第二模型标注结果_512px.csv")
    image_folder = os.path.join(project_dir, "处理后图片")
    
    print(f"使用以下路径:")
    print(f"JSON文件: {json_file}")
    print(f"输出CSV: {output_csv}")
    print(f"图像文件夹: {image_folder}")
    print(f"图像尺寸: 512 × 512 像素")
    
    # 检查图像文件夹是否存在，如果不存在则创建
    if not os.path.exists(image_folder):
        print(f"创建图像文件夹: {image_folder}")
        os.makedirs(image_folder)
    
    # 处理数据
    process_second_model_json(json_file, output_csv, image_folder) 
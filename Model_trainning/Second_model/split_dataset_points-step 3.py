import json
import os
import numpy as np
import pandas as pd

# 加载Label Studio导出的JSON文件
json_file = r"D:\upm_model train\Second_model\project-4-at-2025-06-13-01-11-841a8364.json"
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建图片信息字典
image_info = []
for item in data:
    file_name = item['file_upload']
    patient_id = file_name.split('_')[0]
    
    image_info.append({
        "患者ID": patient_id,
        "图像ID": str(item['id']),
        "文件名": file_name
    })

# 转换为DataFrame便于分析
img_df = pd.DataFrame(image_info)
print(f"总图片数: {len(img_df)}")

# 设置随机种子
np.random.seed(42)

# 随机打乱数据
img_df = img_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 计算训练集、验证集和测试集的大小
total_images = len(img_df)
train_ratio = 0.9
val_ratio = 0.1
test_ratio = 0.0

n_train = int(total_images * train_ratio)
n_val = int(total_images * val_ratio)
n_test = total_images - n_train - n_val

# 划分数据集
train_set = img_df.iloc[:n_train]
val_set = img_df.iloc[n_train:n_train+n_val]
test_set = img_df.iloc[n_train+n_val:]

print(f"\n数据集划分:")
print(f"训练集: {len(train_set)}张图片")
print(f"验证集: {len(val_set)}张图片")
print(f"测试集: {len(test_set)}张图片")

# 为所有图片分配数据集标签
img_df['数据集类型'] = '未使用'

# 保存训练、验证和测试集的图像ID列表
train_img_ids = train_set['图像ID'].tolist()
val_img_ids = val_set['图像ID'].tolist()
test_img_ids = test_set['图像ID'].tolist()

# 为图片分配数据集标签
for i, row in img_df.iterrows():
    if row['图像ID'] in train_img_ids:
        img_df.at[i, '数据集类型'] = '训练集'
    elif row['图像ID'] in val_img_ids:
        img_df.at[i, '数据集类型'] = '验证集'
    elif row['图像ID'] in test_img_ids:
        img_df.at[i, '数据集类型'] = '测试集'

# 分析患者ID分布
train_patients = train_set['患者ID'].unique()
val_patients = val_set['患者ID'].unique()
test_patients = test_set['患者ID'].unique()

print(f"\n患者分布:")
print(f"训练集患者数: {len(train_patients)}")
print(f"验证集患者数: {len(val_patients)}")
print(f"测试集患者数: {len(test_patients)}")

# 检查患者ID重叠情况
train_val_overlap = set(train_patients) & set(val_patients)
train_test_overlap = set(train_patients) & set(test_patients)
val_test_overlap = set(val_patients) & set(test_patients)

if train_val_overlap:
    print(f"警告: 训练集和验证集有{len(train_val_overlap)}个重叠患者ID")
if train_test_overlap:
    print(f"警告: 训练集和测试集有{len(train_test_overlap)}个重叠患者ID")
if val_test_overlap:
    print(f"警告: 验证集和测试集有{len(val_test_overlap)}个重叠患者ID")

# 保存划分信息到CSV文件
output_df = img_df[['患者ID', '图像ID', '文件名', '数据集类型']]
output_df.to_csv(r"D:\upm_model train\Second_model\数据集划分.csv", index=False, encoding="utf-8-sig")

print(f"\n数据集划分信息已保存至：D:\\upm_model train\\Second_model\\数据集划分.csv")

# 创建数据集目录结构
base_dir = r"D:\upm_model train\Second_model\处理后图片"
for dataset_type in ['训练集', '验证集', '测试集']:
    dataset_dir = os.path.join(base_dir, dataset_type)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

print("\n数据集目录结构已创建完成") 
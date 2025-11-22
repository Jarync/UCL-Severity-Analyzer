# 创建新文件 split_dataset.py
import json
import os
import numpy as np
import pandas as pd

# 加载Label Studio导出的JSON文件
json_file = r"D:\upm_model train\唇裂项目\project-1.json"
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建图片信息字典，包含患者ID和术前/术后状态
image_info = []
for item in data:
    file_name = item['file_upload']
    patient_id = file_name.split('_')[0]
    # 判断是术前还是术后
    is_preop = "术前" in file_name or "pre" in file_name.lower()
    type_label = "术前" if is_preop else "术后"
    
    image_info.append({
        "患者ID": patient_id,
        "图像ID": str(item['id']),  # 确保图像ID是字符串类型
        "文件名": file_name,
        "类型": type_label
    })

# 转换为DataFrame便于分析
img_df = pd.DataFrame(image_info)
print(f"总图片数: {len(img_df)}")
print(f"术前图片数: {len(img_df[img_df['类型'] == '术前'])}")
print(f"术后图片数: {len(img_df[img_df['类型'] == '术后'])}")

# 按术前/术后分组
preop_imgs = img_df[img_df['类型'] == '术前']
postop_imgs = img_df[img_df['类型'] == '术后']

# 确定可用的平衡数据量（取术前和术后图片数量的较小值）
n_available = min(len(preop_imgs), len(postop_imgs))
print(f"平衡数据集可用图片数: {n_available * 2}张 (术前: {n_available}张, 术后: {n_available}张)")

#change to 70:15:15 training:validation:test ratio
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# calculate the number of images in the training set, validation set, and test set (ensure even numbers)
n_train = int(n_available * train_ratio * 2)
if n_train % 2 != 0:
    n_train -= 1
    
n_val = int(n_available * val_ratio * 2)
if n_val % 2 != 0:
    n_val -= 1

n_test = int(n_available * test_ratio * 2)
if n_test % 2 != 0:
    n_test -= 1

# ensure the total number does not exceed the available number
while n_train + n_val + n_test > n_available * 2:
    if n_train > n_val and n_train > n_test:
        n_train -= 2
    elif n_val > n_test:
        n_val -= 2
    else:
        n_test -= 2

# 必须确保术前和术后数量相等
n_train_each = n_train // 2
n_val_each = n_val // 2
n_test_each = n_test // 2

print(f"划分比例: 训练集 {train_ratio*100:.0f}%, 验证集 {val_ratio*100:.0f}%, 测试集 {test_ratio*100:.0f}%")
print(f"训练集总数: {n_train}张 (术前: {n_train_each}张, 术后: {n_train_each}张)")
print(f"验证集总数: {n_val}张 (术前: {n_val_each}张, 术后: {n_val_each}张)")
print(f"测试集总数: {n_test}张 (术前: {n_test_each}张, 术后: {n_test_each}张)")

# 设置随机种子
np.random.seed(42)

# 随机选择术前图片
preop_imgs = preop_imgs.sample(frac=1, random_state=42).reset_index(drop=True)
train_preop = preop_imgs.iloc[:n_train_each]
val_preop = preop_imgs.iloc[n_train_each:n_train_each+n_val_each]
test_preop = preop_imgs.iloc[n_train_each+n_val_each:n_train_each+n_val_each+n_test_each]

# 随机选择术后图片
postop_imgs = postop_imgs.sample(frac=1, random_state=42).reset_index(drop=True)
train_postop = postop_imgs.iloc[:n_train_each]
val_postop = postop_imgs.iloc[n_train_each:n_train_each+n_val_each]
test_postop = postop_imgs.iloc[n_train_each+n_val_each:n_train_each+n_val_each+n_test_each]

# 合并训练集、验证集和测试集
train_set = pd.concat([train_preop, train_postop])
val_set = pd.concat([val_preop, val_postop])
test_set = pd.concat([test_preop, test_postop])

# 验证术前术后比例
train_preop_count = len(train_set[train_set['类型'] == '术前'])
train_postop_count = len(train_set[train_set['类型'] == '术后'])
val_preop_count = len(val_set[val_set['类型'] == '术前'])
val_postop_count = len(val_set[val_set['类型'] == '术后'])
test_preop_count = len(test_set[test_set['类型'] == '术前'])
test_postop_count = len(test_set[test_set['类型'] == '术后'])

print("\n数据集验证:")
print(f"训练集: 术前 {train_preop_count}张, 术后 {train_postop_count}张, 比例: {train_preop_count/train_postop_count:.2f}")
print(f"验证集: 术前 {val_preop_count}张, 术后 {val_postop_count}张, 比例: {val_preop_count/val_postop_count:.2f}")
print(f"测试集: 术前 {test_preop_count}张, 术后 {test_postop_count}张, 比例: {test_preop_count/test_postop_count:.2f}")

# 为所有图片分配数据集标签
img_df['数据集类型'] = '未使用'  # 默认值

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

# 统计最终使用和未使用的图片
used_count = len(img_df[img_df['数据集类型'] != '未使用'])
unused_count = len(img_df[img_df['数据集类型'] == '未使用'])
print(f"\n使用图片数: {used_count}张, 未使用图片数: {unused_count}张")

# 分析各类图片的患者ID统计
train_patients = train_set['患者ID'].unique()
val_patients = val_set['患者ID'].unique()
test_patients = test_set['患者ID'].unique()
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
output_df = img_df[['患者ID', '图像ID', '文件名', '类型', '数据集类型']]
output_df.to_csv(r"D:\upm_model train\唇裂项目\数据集划分.csv", index=False, encoding="utf-8-sig")

print(f"\n数据集划分信息已保存至：D:\\upm_model train\\唇裂项目\\数据集划分.csv")
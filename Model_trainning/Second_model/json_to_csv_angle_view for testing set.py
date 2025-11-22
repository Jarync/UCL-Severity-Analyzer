import json
import pandas as pd
import os

# 输入输出路径
json_file = r"D:/upm_model train/project-7angleview.json"
csv_file = r"D:/upm_model train/project-7angleview.csv"

def json_to_csv(json_file, csv_file):
    if not os.path.exists(json_file):
        print(f"找不到JSON文件: {json_file}")
        return
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    for item in data:
        original_filename = item.get('file_upload', '')
        image_id = original_filename.split('_')[0]
        if 'annotations' in item and len(item['annotations']) > 0:
            keypoints = {}
            for point in item['annotations'][0]['result']:
                if point['type'] == 'keypointlabels':
                    label = point['value']['keypointlabels'][0]
                    x = point['value']['x'] / 100
                    y = point['value']['y'] / 100
                    keypoints[label] = (x, y)
            for label, (x, y) in keypoints.items():
                results.append({
                    '图像ID': image_id,
                    '原始文件名': original_filename,
                    '关键点': label,
                    '归一化X': x,
                    '归一化Y': y,
                    '像素X': round(x * 512),
                    '像素Y': round(y * 512)
                })
    if results:
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"已保存为CSV: {csv_file}")
    else:
        print("没有有效数据可保存")

if __name__ == "__main__":
    json_to_csv(json_file, csv_file) 
import json
import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_label_studio_json(json_file_path, output_csv_path, image_folder=None, visualize=True):
    """处理Label Studio导出的JSON文件，计算唇裂等级指标"""
    
    # 检查文件路径是否存在
    if not os.path.exists(json_file_path):
        print(f"错误：找不到JSON文件: {json_file_path}")
        return None
    
    # 检查图像文件夹是否存在
    if image_folder and not os.path.exists(image_folder):
        print(f"警告：找不到图像文件夹: {image_folder}")
        if visualize:
            print("由于找不到图像文件夹，可视化功能将被禁用")
            visualize = False
    
    # 创建可视化文件夹（使用绝对路径）
    project_dir = os.path.dirname(output_csv_path)
    vis_folder = os.path.join(project_dir, "可视化结果_512px")
    if visualize:
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)
            print(f"创建可视化文件夹: {vis_folder}")
    
    # 创建预处理图像文件夹
    preprocessed_folder = os.path.join(project_dir, "预处理图像_512px")
    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder)
        print(f"创建预处理图像文件夹: {preprocessed_folder}")
    
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
        prepost = "术前" if "术前" in original_filename else "术后"
        
        print(f"处理图片: {original_filename} (ID: {image_id}, 类型: {prepost})")
        
        # 尝试加载图像用于可视化
        img = None
        padding_info = {}
        
        if visualize and image_folder and os.path.exists(image_folder):
            for file in os.listdir(image_folder):
                if image_id in file:
                    img_path = os.path.join(image_folder, file)
                    if os.path.exists(img_path):
                        # 读取原始图像
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # 检测填充区域
                        padding_info = detect_padding(img)
                        if padding_info['has_padding']:
                            print(f"  检测到填充: 顶部={padding_info['top']}, 底部={padding_info['bottom']}, 左侧={padding_info['left']}, 右侧={padding_info['right']}")
                            
                            # 去除填充区域
                            img_no_padding = remove_padding(img, padding_info)
                            
                            # 保存无填充图像
                            img_no_padding_rgb = cv2.cvtColor(img_no_padding, cv2.COLOR_RGB2BGR)
                            no_padding_path = os.path.join(preprocessed_folder, f"{image_id}_no_padding.jpg")
                            cv2.imwrite(no_padding_path, img_no_padding_rgb)
                            print(f"  保存无填充图像: {no_padding_path}")
                            
                            # 为了保持原始标注比例正确，仍使用原图及其填充信息
                        break
            
            if img is None:
                print(f"  警告: 在图像文件夹中找不到ID为 {image_id} 的图片")
                img = np.ones((512, 512, 3), dtype=np.uint8) * 255  # 创建白色图像
                padding_info = {'has_padding': False, 'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        if 'annotations' in item and len(item['annotations']) > 0:
            keypoints = {}
            
            # 获取所有关键点
            for point in item['annotations'][0]['result']:
                if point['type'] == 'keypointlabels':
                    label = point['value']['keypointlabels'][0]
                    x = point['value']['x'] / 100  # 归一化到0-1
                    y = point['value']['y'] / 100
                    keypoints[label] = (x, y)
            
            # 如果有所有关键点，计算A和B值
            if len(keypoints) == 6:
                # 获取关键点坐标
                E1 = keypoints['E1']
                E2 = keypoints['E2']
                I1 = keypoints['I1']
                I2 = keypoints['I2']
                N_left = keypoints['N_left']
                N_right = keypoints['N_right']
                
                # 转换为像素坐标用于计算和可视化
                img_size = 512
                
                # 根据填充情况调整坐标
                adjusted_keypoints = {}
                
                if padding_info.get('has_padding', False):
                    # 计算有效图像区域的高度和宽度
                    effective_height = img_size - padding_info['top'] - padding_info['bottom']
                    effective_width = img_size - padding_info['left'] - padding_info['right']
                    
                    # 调整所有关键点坐标
                    for label, (x, y) in keypoints.items():
                        # 移除填充影响，将坐标重新缩放到有效区域
                        adj_x = (x * img_size - padding_info['left']) / effective_width if effective_width > 0 else x
                        adj_y = (y * img_size - padding_info['top']) / effective_height if effective_height > 0 else y
                        adjusted_keypoints[label] = (adj_x, adj_y)
                    
                    # 更新关键点坐标
                    E1 = adjusted_keypoints.get('E1', E1)
                    E2 = adjusted_keypoints.get('E2', E2)
                    I1 = adjusted_keypoints.get('I1', I1)
                    I2 = adjusted_keypoints.get('I2', I2)
                    N_left = adjusted_keypoints.get('N_left', N_left)
                    N_right = adjusted_keypoints.get('N_right', N_right)
                    
                    print(f"  坐标已调整以补偿填充区域")
                    for label, (x, y) in keypoints.items():
                        adj_x, adj_y = adjusted_keypoints[label]
                        print(f"    {label}: 原始=({x:.4f}, {y:.4f}), 调整后=({adj_x:.4f}, {adj_y:.4f})")
                
                # 1. 计算A值 - 左右鼻翼沟垂直距离
                A = abs(N_left[1] - N_right[1]) * img_size
                
                # 2. 计算中心点
                # 计算E1和E2的中点
                mid_E_x = (E1[0] + E2[0]) / 2
                mid_E_y = (E1[1] + E2[1]) / 2
                
                # 计算I1和I2的中点
                mid_I_x = (I1[0] + I2[0]) / 2
                mid_I_y = (I1[1] + I2[1]) / 2
                
                # 计算中心点
                center_x = (mid_E_x + mid_I_x) / 2
                center_y = (mid_E_y + mid_I_y) / 2
                
                # 3. 确定更低的鼻翼沟点
                lowest_nostril = N_left if N_left[1] > N_right[1] else N_right
                lowest_nostril_label = "N_left" if N_left[1] > N_right[1] else "N_right"
                
                # 计算鼻翼基底Y值（使用最低点而不是平均值）
                nostril_base_y = lowest_nostril[1]
                
                # 4. 计算B值 - 中心点到鼻翼基底的垂直距离
                B = abs(center_y - nostril_base_y) * img_size
                
                # 5. 计算比率
                ratio = A / B if B != 0 else 0
                
                # 6. 等级判定
                if ratio <= 0.05:
                    grade = "轻度 (3分)"
                elif ratio <= 0.10:
                    grade = "中度 (2分)"
                else:
                    grade = "重度 (1分)"
                
                print(f"  计算结果: A={A:.2f}, B={B:.2f}, A/B={ratio:.4f}, 等级={grade}")
                print(f"  原始坐标: N_left=({N_left[0]:.4f}, {N_left[1]:.4f}), N_right=({N_right[0]:.4f}, {N_right[1]:.4f})")
                print(f"  中心点: ({center_x:.4f}, {center_y:.4f})")
                print(f"  鼻翼基底: 使用{lowest_nostril_label} ({lowest_nostril[0]:.4f}, {lowest_nostril[1]:.4f})")
                print(f"  A计算: |{N_left[1]:.4f} - {N_right[1]:.4f}| × {img_size} = {A:.2f}")
                print(f"  B计算: |{center_y:.4f} - {nostril_base_y:.4f}| × {img_size} = {B:.2f}")
                
                # 可视化
                if visualize:
                    # 创建两个可视化：原始图像和无填充图像
                    for show_padding in [True, False]:
                        # 转换为像素坐标
                        display_img = img.copy()
                        points_img = img.copy() if show_padding else (
                            remove_padding(img, padding_info) if padding_info.get('has_padding', False) else img.copy()
                        )
                        
                        # 使用当前图像的尺寸
                        h, w = points_img.shape[:2]
                        
                        # 像素坐标转换，如果不显示填充，则使用调整后的坐标
                        if not show_padding and padding_info.get('has_padding', False):
                            # 使用调整后的坐标
                            E1_px = (int(adjusted_keypoints['E1'][0] * w), int(adjusted_keypoints['E1'][1] * h))
                            E2_px = (int(adjusted_keypoints['E2'][0] * w), int(adjusted_keypoints['E2'][1] * h))
                            I1_px = (int(adjusted_keypoints['I1'][0] * w), int(adjusted_keypoints['I1'][1] * h))
                            I2_px = (int(adjusted_keypoints['I2'][0] * w), int(adjusted_keypoints['I2'][1] * h))
                            N_left_px = (int(adjusted_keypoints['N_left'][0] * w), int(adjusted_keypoints['N_left'][1] * h))
                            N_right_px = (int(adjusted_keypoints['N_right'][0] * w), int(adjusted_keypoints['N_right'][1] * h))
                            
                            # 中心点计算
                            mid_E_px = (int((adjusted_keypoints['E1'][0] + adjusted_keypoints['E2'][0])/2 * w), 
                                        int((adjusted_keypoints['E1'][1] + adjusted_keypoints['E2'][1])/2 * h))
                            mid_I_px = (int((adjusted_keypoints['I1'][0] + adjusted_keypoints['I2'][0])/2 * w), 
                                        int((adjusted_keypoints['I1'][1] + adjusted_keypoints['I2'][1])/2 * h))
                            center_px = (int((mid_E_px[0] + mid_I_px[0])/2), int((mid_E_px[1] + mid_I_px[1])/2))
                            
                            # 最低鼻翼点
                            lowest_nostril_px = N_left_px if N_left[1] > N_right[1] else N_right_px
                        else:
                            # 使用原始坐标
                            E1_px = (int(E1[0] * w), int(E1[1] * h))
                            E2_px = (int(E2[0] * w), int(E2[1] * h))
                            I1_px = (int(I1[0] * w), int(I1[1] * h))
                            I2_px = (int(I2[0] * w), int(I2[1] * h))
                            N_left_px = (int(N_left[0] * w), int(N_left[1] * h))
                            N_right_px = (int(N_right[0] * w), int(N_right[1] * h))
                            
                            # 中心点和中间点可视化
                            mid_E_px = (int(mid_E_x * w), int(mid_E_y * h))
                            mid_I_px = (int(mid_I_x * w), int(mid_I_y * h))
                            center_px = (int(center_x * w), int(center_y * h))
                            
                            # 鼻翼基底点（使用最低鼻翼点）
                            lowest_nostril_px = (int(lowest_nostril[0] * w), int(lowest_nostril[1] * h))
                        
                        # 创建图像
                        plt.figure(figsize=(10, 10))
                        plt.imshow(points_img)
                        
                        # 如果显示填充，标记填充区域
                        if show_padding and padding_info.get('has_padding', False):
                            if padding_info['top'] > 0:
                                plt.axhspan(0, padding_info['top'], color='lightblue', alpha=0.3)
                                plt.text(10, padding_info['top']/2, "白色填充区域", color='blue', fontsize=12)
                            if padding_info['bottom'] > 0:
                                plt.axhspan(h-padding_info['bottom'], h, color='lightblue', alpha=0.3)
                                plt.text(10, h-padding_info['bottom']/2, "白色填充区域", color='blue', fontsize=12)
                            if padding_info['left'] > 0:
                                plt.axvspan(0, padding_info['left'], color='lightblue', alpha=0.3)
                                plt.text(padding_info['left']/2, 30, "白色填充区域", color='blue', fontsize=12, rotation=90)
                            if padding_info['right'] > 0:
                                plt.axvspan(w-padding_info['right'], w, color='lightblue', alpha=0.3)
                                plt.text(w-padding_info['right']/2, 30, "白色填充区域", color='blue', fontsize=12, rotation=90)
                        
                        # 绘制关键点
                        plt.plot(E1_px[0], E1_px[1], 'ro', markersize=8)
                        plt.plot(E2_px[0], E2_px[1], 'ro', markersize=8)
                        plt.text(E1_px[0]+5, E1_px[1]-5, 'E1', color='red', fontsize=12)
                        plt.text(E2_px[0]+5, E2_px[1]-5, 'E2', color='red', fontsize=12)
                        
                        plt.plot(I1_px[0], I1_px[1], 'go', markersize=8)
                        plt.plot(I2_px[0], I2_px[1], 'go', markersize=8)
                        plt.text(I1_px[0]+5, I1_px[1]-5, 'I1', color='green', fontsize=12)
                        plt.text(I2_px[0]+5, I2_px[1]-5, 'I2', color='green', fontsize=12)
                        
                        plt.plot(N_left_px[0], N_left_px[1], 'bo', markersize=8)
                        plt.plot(N_right_px[0], N_right_px[1], 'bo', markersize=8)
                        plt.text(N_left_px[0]+5, N_left_px[1]-5, 'N_L', color='blue', fontsize=12)
                        plt.text(N_right_px[0]+5, N_right_px[1]-5, 'N_R', color='blue', fontsize=12)
                        
                        # 绘制中心点计算过程
                        plt.plot(mid_E_px[0], mid_E_px[1], 'mo', markersize=6)
                        plt.text(mid_E_px[0]+5, mid_E_px[1]-5, 'mid_E', color='magenta', fontsize=10)
                        
                        plt.plot(mid_I_px[0], mid_I_px[1], 'mo', markersize=6)
                        plt.text(mid_I_px[0]+5, mid_I_px[1]-5, 'mid_I', color='magenta', fontsize=10)
                        
                        # 用黄色点标记最终中心点
                        plt.plot(center_px[0], center_px[1], 'yo', markersize=8)
                        plt.text(center_px[0]+5, center_px[1]-5, 'Center', color='yellow', fontsize=12)
                        
                        # 标记最低鼻翼点
                        plt.plot(lowest_nostril_px[0], lowest_nostril_px[1], 'co', markersize=8, alpha=0.7)
                        plt.text(lowest_nostril_px[0]+5, lowest_nostril_px[1]+5, f'最低点({lowest_nostril_label})', color='cyan', fontsize=10)
                        
                        # 绘制A线 - 鼻翼沟垂直距离
                        if N_left[1] > N_right[1]:
                            plt.plot([N_right_px[0], N_right_px[0]], [N_right_px[1], N_left_px[1]], 'r-', linewidth=2)
                            plt.text(N_right_px[0]+10, (N_right_px[1]+N_left_px[1])/2, f'A={A:.1f}', color='red', fontsize=12)
                        else:
                            plt.plot([N_left_px[0], N_left_px[0]], [N_left_px[1], N_right_px[1]], 'r-', linewidth=2)
                            plt.text(N_left_px[0]+10, (N_left_px[1]+N_right_px[1])/2, f'A={A:.1f}', color='red', fontsize=12)
                        
                        # 绘制B线 - 中心点到鼻翼基底的垂直距离
                        plt.plot([center_px[0], center_px[0]], [center_px[1], lowest_nostril_px[1]], 'g-', linewidth=2)
                        plt.text(center_px[0]+10, (center_px[1]+lowest_nostril_px[1])/2, f'B={B:.1f}', color='green', fontsize=12)
                        
                        # 绘制水平辅助线标示鼻翼基底Y位置
                        plt.plot([0, w], [lowest_nostril_px[1], lowest_nostril_px[1]], 'c--', linewidth=1, alpha=0.5)
                        
                        # 添加标题和信息
                        padding_status = "（有填充）" if show_padding and padding_info.get('has_padding', False) else "（无填充）"
                        plt.title(f"{prepost} - A/B = {ratio:.4f} - {grade} {padding_status}")
                        plt.text(10, 30, f"A = {A:.2f} (鼻翼沟垂直差异)", color='red', fontsize=15)
                        plt.text(10, 60, f"B = {B:.2f} (中心点至最低鼻翼沟垂直距离)", color='green', fontsize=15)
                        plt.text(10, 90, f"A/B = {ratio:.4f}", color='blue', fontsize=15)
                        plt.text(10, 120, f"评级: {grade}", color='purple', fontsize=15)
                        
                        # 如果检测到填充，显示信息
                        if padding_info.get('has_padding', False):
                            plt.text(10, 150, f"检测到白色填充区域: 上={padding_info['top']}, 下={padding_info['bottom']}, 左={padding_info['left']}, 右={padding_info['right']}", 
                                     color='blue', fontsize=12)
                        
                        # 保存图像
                        suffix = "_with_padding" if show_padding and padding_info.get('has_padding', False) else "_no_padding"
                        output_path = os.path.join(vis_folder, f"{image_id}_{prepost}{suffix}_analysis.jpg")
                        plt.savefig(output_path)
                        plt.close()
                        print(f"  可视化结果已保存至: {output_path}")
                
                # 添加结果
                for label, (x, y) in keypoints.items():
                    # 如果有调整后的坐标，同时保存原始和调整后的坐标
                    adj_x, adj_y = adjusted_keypoints.get(label, (x, y)) if padding_info.get('has_padding', False) else (x, y)
                    
                    results.append({
                        '图像ID': image_id,
                        '原始文件名': original_filename,
                        '关键点': label,
                        '归一化X': x,
                        '归一化Y': y,
                        '调整后X': adj_x,
                        '调整后Y': adj_y,
                        '像素X': round(x * img_size),
                        '像素Y': round(y * img_size),
                        '术前/术后': prepost,
                        'A值': round(A, 2),
                        'B值': round(B, 2),
                        'A/B比值': round(ratio, 4),
                        '唇裂等级': grade,
                        '有填充': padding_info.get('has_padding', False),
                        '填充_上': padding_info.get('top', 0),
                        '填充_下': padding_info.get('bottom', 0),
                        '填充_左': padding_info.get('left', 0),
                        '填充_右': padding_info.get('right', 0)
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
        preop_count = len(set(df[df['术前/术后'] == '术前']['图像ID']))
        postop_count = len(set(df[df['术前/术后'] == '术后']['图像ID']))
        
        print(f"\n统计信息:")
        print(f"总标注图片数: {image_count}")
        print(f"术前图片数: {preop_count}")
        print(f"术后图片数: {postop_count}")
        
        # 生成术前术后对比
        compare_prepost(df)
        
        return df
    else:
        print("警告: 没有处理任何有效数据")
        return None

def detect_padding(img):
    """检测图像中的填充区域"""
    if img is None:
        return {'has_padding': False, 'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    
    h, w = img.shape[:2]
    padding = {'has_padding': False, 'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    
    # 检测顶部填充
    for y in range(h):
        row_mean = np.mean(img[y, :, :])
        if row_mean < 240:  # 不是白色
            padding['top'] = y
            break
    
    # 检测底部填充
    for y in range(h-1, -1, -1):
        row_mean = np.mean(img[y, :, :])
        if row_mean < 240:  # 不是白色
            padding['bottom'] = h - y - 1
            break
    
    # 检测左侧填充
    for x in range(w):
        col_mean = np.mean(img[:, x, :])
        if col_mean < 240:  # 不是白色
            padding['left'] = x
            break
    
    # 检测右侧填充
    for x in range(w-1, -1, -1):
        col_mean = np.mean(img[:, x, :])
        if col_mean < 240:  # 不是白色
            padding['right'] = w - x - 1
            break
    
    # 如果任何一个方向有填充，则设置has_padding为True
    if padding['top'] > 0 or padding['bottom'] > 0 or padding['left'] > 0 or padding['right'] > 0:
        padding['has_padding'] = True
    
    return padding

def remove_padding(img, padding_info):
    """根据填充信息去除图像填充"""
    if not padding_info.get('has_padding', False) or img is None:
        return img
    
    h, w = img.shape[:2]
    top = padding_info['top']
    bottom = padding_info['bottom']
    left = padding_info['left']
    right = padding_info['right']
    
    # 裁剪图像以去除填充
    if top >= h or bottom >= h or left >= w or right >= w or top + bottom >= h or left + right >= w:
        return img  # 防止裁剪参数无效
    
    return img[top:h-bottom, left:w-right] if top < h and bottom < h and left < w and right < w else img

def compare_prepost(df):
    """生成术前术后对比报告"""
    # 获取所有图像ID
    image_ids = set()
    for img_id in df['图像ID']:
        # 检查是否同时有术前和术后
        pre = df[(df['图像ID'] == img_id) & (df['术前/术后'] == '术前')]
        post = df[(df['图像ID'] == img_id) & (df['术前/术后'] == '术后')]
        if len(pre) > 0 and len(post) > 0:
            image_ids.add(img_id)
    
    if not image_ids:
        print("没有找到可以对比的术前/术后图片")
        return
    
    # 准备对比数据
    compare_data = []
    
    for img_id in image_ids:
        pre_data = df[(df['图像ID'] == img_id) & (df['术前/术后'] == '术前')]
        post_data = df[(df['图像ID'] == img_id) & (df['术前/术后'] == '术后')]
        
        if len(pre_data) > 0 and len(post_data) > 0:
            pre_ratio = pre_data['A/B比值'].iloc[0]
            post_ratio = post_data['A/B比值'].iloc[0]
            improvement = pre_ratio - post_ratio
            
            compare_data.append({
                '图像ID': img_id,
                '术前比值': pre_ratio,
                '术后比值': post_ratio,
                '改善值': improvement,
                '术前等级': pre_data['唇裂等级'].iloc[0],
                '术后等级': post_data['唇裂等级'].iloc[0]
            })
    
    # 保存对比结果
    if compare_data:
        compare_df = pd.DataFrame(compare_data)
        compare_df.to_csv("术前术后对比结果.csv", index=False, encoding='utf-8-sig')
        print("术前术后对比结果已保存至: 术前术后对比结果.csv")
        
        # 输出对比摘要
        for row in compare_data:
            print(f"图像ID: {row['图像ID']}")
            print(f"  术前: {row['术前比值']:.4f} ({row['术前等级']})")
            print(f"  术后: {row['术后比值']:.4f} ({row['术后等级']})")
            print(f"  改善: {row['改善值']:.4f}")
            print()

# 主程序执行 - 修改后的版本
if __name__ == "__main__":
    # 修正文件路径
    json_file = r"D:\upm_model train\唇裂项目\project-1.json"  # 使用用户提供的JSON文件路径
    project_dir = r"D:\upm_model train\唇裂项目"
    output_csv = os.path.join(project_dir, "唇裂标注分析结果_512px.csv")
    image_folder = os.path.join(project_dir, "处理后图片")
    
    # 创建可视化结果文件夹（使用绝对路径）
    vis_folder = os.path.join(project_dir, "可视化结果_512px")
    if not os.path.exists(vis_folder):
        print(f"创建可视化结果文件夹: {vis_folder}")
        os.makedirs(vis_folder)
    
    # 创建专门用于512×512图像的预处理文件夹
    preprocessed_folder = os.path.join(project_dir, "预处理图像_512px")
    if not os.path.exists(preprocessed_folder):
        print(f"创建预处理图像文件夹: {preprocessed_folder}")
        os.makedirs(preprocessed_folder)
    
    print(f"使用以下路径:")
    print(f"JSON文件: {json_file}")
    print(f"输出CSV: {output_csv}")
    print(f"图像文件夹: {image_folder}")
    print(f"可视化结果: {vis_folder}")
    print(f"图像尺寸: 512 × 512 像素")
    
    # 检查图像文件夹是否存在，如果不存在则创建
    if not os.path.exists(image_folder):
        print(f"创建图像文件夹: {image_folder}")
        os.makedirs(image_folder)
    
    # 开启可视化功能
    process_label_studio_json(json_file, output_csv, image_folder, visualize=True)
    
    print("\n处理完成，请在以下位置查看可视化结果:")
    print(vis_folder) 
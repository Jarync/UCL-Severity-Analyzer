# lib/datasets/cleftlip.py
# ------------------------------------------------------------------
# 6-landmark cleft-lip dataset for HRNet – drop-in replacement
# ------------------------------------------------------------------
import os, warnings, numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# HRNet 原生生成热图函数
from ..utils.transforms import generate_target

class CleftLip(Dataset):
    """
    ROOT/
        处理后图片/
        唇裂标注分析结果_512px.csv
        数据集划分.csv
    """
    NUM_JOINTS = 6
    SIGMA      = 2.0                       # 设为合理的适中值
    KP_NAMES   = ['E1','E2','I1','I2','N_left','N_right']

    COL = {                                 # 中文列名映射
        'SPLIT_ID'   : '图像ID',
        'SPLIT_TYPE' : '数据集类型',
        'IMG_ID'     : '图像ID',
        'FILENAME'   : '原始文件名',
        'KPT_NAME'   : '关键点',
        'X'          : '归一化X',
        'Y'          : '归一化Y',
    }

    # --------------------------------------------------------------
    def __init__(self, cfg, is_train=True, transform=None):
        split_tag   = '训练集' if is_train else '验证集'
        root        = cfg.DATASET.ROOT.rstrip('/\\')

        anno_csv    = os.path.join(root, cfg.DATASET.TRAINSET)
        split_csv   = os.path.join(root, cfg.DATASET.TESTSET)
        self.img_dir= os.path.join(root, cfg.DATASET.IMAGE_DIR)

        # 从配置读取图像尺寸，而不是硬编码
        self.IMAGE_SIZE = tuple(int(x) for x in cfg.MODEL.IMAGE_SIZE)

        # 读取 CSV 并去掉空格
        strip = lambda df: df.applymap(lambda x: x.strip() if isinstance(x,str) else x)
        self.anno_df  = strip(pd.read_csv(anno_csv, dtype=str))
        self.split_df = strip(pd.read_csv(split_csv, dtype=str))

        # ------------------------------------------------------------------
        # 建 ID → 文件名 映射
        id2file = {}
        if self.COL['FILENAME'] in self.split_df.columns:
            id2file.update(
                self.split_df.dropna(subset=[self.COL['FILENAME']])
                .set_index(self.COL['SPLIT_ID'])[self.COL['FILENAME']].to_dict()
            )
        id2file.update(                           # 用 anno_df 补全缺失
            self.anno_df.dropna(subset=[self.COL['FILENAME']])
            .drop_duplicates(subset=[self.COL['IMG_ID']])
            .set_index(self.COL['IMG_ID'])[self.COL['FILENAME']].to_dict()
        )
        # ------------------------------------------------------------------
        # 当前 split 的所有样本
        split_ids = (
            self.split_df[self.split_df[self.COL['SPLIT_TYPE']] == split_tag]
            [self.COL['SPLIT_ID']].tolist()
        )

        # 读取 cfg.MODEL.HEATMAP_SIZE（保证与网络一致）
        self.HEATMAP_SIZE = tuple(int(x) for x in cfg.MODEL.HEATMAP_SIZE)

        # 构建样本列表
        self.samples = []
        for img_id in tqdm(split_ids, desc=f"加载{split_tag}数据"):
            fname = id2file.get(img_id, '')
            if not fname:               # 没有文件名
                continue

            # ------- 关键点 -------
            grp = self.anno_df[self.anno_df[self.COL['FILENAME']] == fname]
            if grp.empty:
                continue
            kp = np.full((self.NUM_JOINTS, 2), -1, np.float32)
            for i, name in enumerate(self.KP_NAMES):
                row = grp[grp[self.COL['KPT_NAME']] == name]
                if len(row):
                    kp[i] = [float(row[self.COL['X']].iloc[0]),
                             float(row[self.COL['Y']].iloc[0])]

            # ------- 定位图片 -------
            img_path = os.path.join(self.img_dir, fname)
            if not os.path.isfile(img_path):
                tail = fname.split('-', 1)[-1]
                hits = [f for f in os.listdir(self.img_dir)
                        if f.lower().endswith(tail.lower())]
                if hits:
                    img_path = os.path.join(self.img_dir, hits[0])

            if os.path.isfile(img_path):
                self.samples.append((img_path, kp))
            else:
                warnings.warn(f'⚠️ 找不到图片: {img_path}  — 已跳过')

        if not self.samples:
            raise RuntimeError('💥 数据集初始化后仍为空，请检查路径/文件名！')

        # 图像 transform
        self.transform = transform or T.Compose([
            T.Resize(self.IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    # --------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, kp_norm = self.samples[idx]

        # -------- 图像 --------
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # -------- 热图 --------
        target = np.zeros((self.NUM_JOINTS, *self.HEATMAP_SIZE), np.float32)
        for j in range(self.NUM_JOINTS):
            x_norm, y_norm = kp_norm[j]
            if x_norm < 0 or y_norm < 0:
                continue                      # 缺失点
            # 明确指定x和y对应宽度和高度，避免HEATMAP_SIZE格式变化导致的坐标混淆
            # 修正：确保x使用width(W)，y使用height(H)
            heatmap_width, heatmap_height = self.HEATMAP_SIZE[1], self.HEATMAP_SIZE[0]  # W, H
            pt = np.array([x_norm * heatmap_width,  # x ↔ width (W)
                           y_norm * heatmap_height]) # y ↔ height (H)
            target[j] = generate_target(target[j], pt, self.SIGMA)

        target = torch.from_numpy(target)

        # 将归一化坐标转换为像素坐标，与preds坐标系统匹配
        img_width, img_height = self.IMAGE_SIZE[1], self.IMAGE_SIZE[0]  # W, H
        pix_pts = kp_norm * np.array([img_width, img_height])   # (6,2) 归一化 → 像素坐标
        
        # 修正meta里的center和scale，确保正确表示原图到热图的转换关系
        # meta 里保存像素坐标，与preds匹配，给 NME 计算用
        meta = {
            'index' : idx,
            'pts'   : torch.from_numpy(pix_pts).float()      # 使用像素坐标
        }
        return img, target, meta

def visualize_keypoints(image, keypoints, predictions=None, save_path=None, is_normalized=True, image_size=(512, 512)):
    """可视化关键点预测结果
    
    参数:
        image: 图像张量或PIL图像
        keypoints: 关键点坐标 [num_keypoints, 2] 
        predictions: 预测关键点坐标 [num_keypoints, 2]
        save_path: 保存路径，若为None则显示图像
        is_normalized: 输入坐标是否为归一化坐标(0-1)，若为False则为像素坐标
        image_size: 图像尺寸，用于归一化/反归一化坐标，默认为(512, 512)
    """
    try:
        # 如果是张量，转换为numpy并进行正确的反归一化
        if isinstance(image, torch.Tensor):
            # 首先移动到CPU并转换为numpy数组
            image = image.cpu().numpy().transpose(1, 2, 0)
            
            # 反归一化 - 恢复原始图像显示效果
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            
            # 裁剪值到0-1范围
            image = np.clip(image, 0, 1)
            
            # 转换为0-255的RGB图像
            image = (image * 255).astype(np.uint8)
        
        # 创建matplotlib图形
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # 定义关键点名称和颜色
        keypoint_names = ['E1', 'E2', 'I1', 'I2', 'N_left', 'N_right']
        colors = ['r', 'r', 'g', 'g', 'b', 'b']
        
        # 转换关键点坐标至像素坐标
        h, w = image.shape[:2]
        img_w, img_h = image_size
        
        # 坐标系转换：确保处理像素坐标
        if keypoints is not None:
            keypoints = keypoints.cpu().numpy() if isinstance(keypoints, torch.Tensor) else keypoints
            # 如果是归一化坐标，转换为像素坐标
            if is_normalized:
                keypoints = keypoints.copy() * np.array([img_w, img_h])
        
        if predictions is not None:
            predictions = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
            # 如果predictions是像素坐标而需要归一化坐标，或反之，进行转换
            if is_normalized:
                predictions = predictions.copy() * np.array([img_w, img_h])
        
        # 绘制真实关键点 - 使用实心大圆点
        if keypoints is not None:
            for i, (kp, name, color) in enumerate(zip(keypoints, keypoint_names, colors)):
                x, y = int(kp[0] * w / img_w), int(kp[1] * h / img_h)  # 调整到实际图像大小
                plt.scatter(x, y, c=color, s=40, marker='o', alpha=0.7)
                plt.text(x+5, y+5, name, color=color, fontsize=12, weight='bold')
        
        # 绘制预测关键点 - 使用X形，更容易与真实值区分
        if predictions is not None:
            for i, (kp, name) in enumerate(zip(predictions, keypoint_names)):
                x, y = int(kp[0] * w / img_w), int(kp[1] * h / img_h)  # 调整到实际图像大小
                plt.scatter(x, y, c='yellow', s=30, marker='x', linewidths=2)
                plt.text(x-15, y-15, f'pred_{name}', color='yellow', fontsize=10, weight='bold')
                
                # 添加连线显示预测点与真实点之间的偏差
                if keypoints is not None:
                    true_x, true_y = int(keypoints[i][0] * w / img_w), int(keypoints[i][1] * h / img_h)
                    plt.plot([x, true_x], [y, true_y], 'y--', alpha=0.7, linewidth=1)
                    
                    # 计算并显示欧氏距离
                    dist = np.sqrt((x-true_x)**2 + (y-true_y)**2)
                    mid_x, mid_y = (x + true_x) // 2, (y + true_y) // 2
                    plt.text(mid_x, mid_y, f'{dist:.1f}px', color='white', fontsize=8, 
                             bbox=dict(facecolor='black', alpha=0.5))
        
        # 添加图例
        plt.scatter([], [], c='r', s=40, marker='o', alpha=0.7, label='真实眼部关键点(E1,E2)')
        plt.scatter([], [], c='g', s=40, marker='o', alpha=0.7, label='真实内嘴角关键点(I1,I2)')
        plt.scatter([], [], c='b', s=40, marker='o', alpha=0.7, label='真实鼻翼关键点(N_left,N_right)')
        plt.scatter([], [], c='yellow', s=30, marker='x', label='预测关键点')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # 添加标题显示准确率信息
        if predictions is not None and keypoints is not None:
            # 计算像素坐标下的欧氏距离
            distances = []
            for i in range(len(keypoints)):
                true_x, true_y = int(keypoints[i][0] * w / img_w), int(keypoints[i][1] * h / img_h)
                pred_x, pred_y = int(predictions[i][0] * w / img_w), int(predictions[i][1] * h / img_h)
                dist = np.sqrt((pred_x-true_x)**2 + (pred_y-true_y)**2)
                distances.append(dist)
            
            avg_dist = np.mean(distances)
            plt.title(f'平均关键点误差: {avg_dist:.1f}像素', fontsize=14)
        
        # 保存或显示图像
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"可视化过程中出错: {str(e)}")
        # 出错时不中断训练流程

def calculate_accuracy(predictions, targets, threshold=0.05, is_normalized=True, image_size=(512, 512)):
    """计算关键点检测准确率
    
    参数:
        predictions: 预测关键点坐标 [batch_size, num_keypoints, 2] 
        targets: 真实关键点坐标 [batch_size, num_keypoints, 2]
        threshold: 正确预测的阈值，默认为0.05（归一化空间中距离）
        is_normalized: 输入坐标是否为归一化坐标(0-1)，若为False则为像素坐标
        image_size: 图像尺寸，用于归一化坐标，默认为(512, 512)
    
    返回:
        准确率百分比
    """
    # 确保输入是numpy数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()
    
    # 如果是像素坐标，需要归一化
    if not is_normalized:
        img_w, img_h = image_size
        predictions = predictions.copy() / np.array([img_w, img_h])
        targets = targets.copy() / np.array([img_w, img_h])
    
    # 计算归一化空间的欧氏距离
    distances = np.sqrt(np.sum((predictions - targets) ** 2, axis=2))
    
    # 准确率 - 定义为归一化距离小于阈值的关键点百分比
    accuracy = np.mean(distances < threshold) * 100
    
    return accuracy

import os
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 设置后端为非交互式
import matplotlib.pyplot as plt
import random

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeypointDataset(Dataset):
    """鼻孔关键点数据集"""
    NUM_JOINTS = 5  # CC1, CC2, CN1, CN2, N
    SIGMA = 3.0  # 调整高斯核大小到3.0
    IMAGE_SIZE = [256, 256]  # 输入图像尺寸，与配置文件一致
    HEATMAP_SIZE = [64, 64]  # 热图尺寸，与配置文件一致
    
    def __init__(self, csv_file, split_file, img_dir, is_train=True, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        
        # 读取CSV文件
        self.df = pd.read_csv(csv_file)
        self.split_df = pd.read_csv(split_file)
        
        # 调试信息
        logger.info(f'标注CSV文件中的总行数: {len(self.df)}')
        logger.info(f'数据集划分CSV文件中的总行数: {len(self.split_df)}')
        
        # 获取训练/验证集的图片ID
        split_tag = '训练集' if is_train else '验证集'
        logger.info(f'正在加载{split_tag}数据...')
        
        # 从划分文件中获取患者ID，并提取后半部分
        split_ids = self.split_df[self.split_df['数据集类型'] == split_tag]['患者ID'].tolist()
        self.valid_ids = []
        for full_id in split_ids:
            # 从患者ID中提取后半部分（例如从876ce5cb-54eb29d3中提取54eb29d3）
            short_id = full_id.split('-')[1]
            self.valid_ids.append(short_id)
        
        logger.info(f'在划分文件中找到的{split_tag}患者ID数量: {len(self.valid_ids)}')
        
        # 创建图片文件名映射
        self.filename_map = {}
        for img_path in os.listdir(self.img_dir):
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 从文件名中提取ID部分（例如从54eb29d3_IMGA0034.JPG中提取54eb29d3）
                id_part = img_path.split('_')[0]
                self.filename_map[id_part] = img_path
        
        logger.info(f'图片目录中的文件数量: {len(self.filename_map)}')
        
        # 检查每个图像是否有完整的5个关键点
        self.samples = []
        required_points = ['CC1', 'CC2', 'CN1', 'CN2', 'N']
        
        for short_id in self.valid_ids:
            # 在标注文件中查找包含这个短ID的所有图像ID
            matching_rows = self.df[self.df['图像ID'].str.contains(short_id, na=False)]
            if len(matching_rows) > 0:
                # 获取该患者的所有图像ID
                image_ids = matching_rows['图像ID'].unique()
                for image_id in image_ids:
                    # 对每个图像ID检查是否有完整的关键点
                    img_data = self.df[self.df['图像ID'] == image_id]
                    points = img_data['关键点'].unique()
                    if all(point in points for point in required_points):
                        # 从图像ID中提取短ID
                        img_short_id = image_id.split('-')[1]
                        if img_short_id in self.filename_map:
                            self.samples.append((image_id, img_short_id))
                        else:
                            logger.warning(f'找不到图像ID {image_id} 对应的图片文件')
                    else:
                        logger.warning(f'图像ID {image_id} 缺少部分关键点: {set(required_points) - set(points)}')
            else:
                logger.warning(f'在标注文件中找不到包含ID {short_id} 的图像')
        
        logger.info(f'最终有效的{split_tag}样本数量: {len(self.samples)}')
        
        if len(self.samples) == 0:
            logger.error('没有找到有效的样本！请检查以下几点：')
            logger.error('1. 数据集划分文件中的"数据集类型"列是否包含"训练集"和"验证集"')
            logger.error('2. 标注CSV文件中的图像ID是否与划分文件中的患者ID匹配')
            logger.error('3. 图片目录中的文件名格式是否正确')
            logger.error('4. 是否所有图片都有完整的5个关键点标注')
        
        # define data augmentation
        if self.is_train:
            self.augment = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
            ])
        else:
            self.augment = None
    
    def __len__(self):
        return len(self.samples) * (2 if self.is_train else 1)  # return original image and augmented image
    
    def generate_target(self, points):
        """生成热图目标"""
        target = np.zeros((self.NUM_JOINTS, *self.HEATMAP_SIZE), dtype=np.float32)
        
        for i, (x_norm, y_norm) in enumerate(points):
            if x_norm < 0 or y_norm < 0:  # 跳过无效点
                continue
                
            # 明确指定height和width
            heatmap_h, heatmap_w = self.HEATMAP_SIZE  # H, W
            
            # 转换到热图尺度
            cx = x_norm * heatmap_w  # x ↔ width
            cy = y_norm * heatmap_h  # y ↔ height
            
            # 生成高斯热图
            tmp = int(self.SIGMA * 3)  # 确保是整数
            mu_x, mu_y = int(cx + 0.5), int(cy + 0.5)
            
            # 计算热图范围 [x_min, y_min], [x_max, y_max]
            ul = [int(mu_x - tmp), int(mu_y - tmp)]  # 确保是整数
            br = [int(mu_x + tmp + 1), int(mu_y + tmp + 1)]  # 确保是整数
            
            # 如果关键点在图像外，跳过
            if ul[0] >= heatmap_w or ul[1] >= heatmap_h or br[0] < 0 or br[1] < 0:
                continue
            
            # 生成高斯核
            size = 2 * tmp + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]
            g = np.exp(-((x - tmp) ** 2 + (y - tmp) ** 2) / (2 * self.SIGMA ** 2))
            
            # 裁剪范围 - 确保所有索引都是整数
            g_x = (int(max(0, -ul[0])), int(min(br[0], heatmap_w) - ul[0]))
            g_y = (int(max(0, -ul[1])), int(min(br[1], heatmap_h) - ul[1]))
            img_x = (int(max(0, ul[0])), int(min(br[0], heatmap_w)))
            img_y = (int(max(0, ul[1])), int(min(br[1], heatmap_h)))
            
            target[i][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        
        return torch.from_numpy(target)
    
    def __getitem__(self, idx):
        # 对于训练集，idx需要映射到实际样本
        real_idx = idx // 2 if self.is_train else idx
        is_augmented = self.is_train and (idx % 2 == 1)
        
        image_id, short_id = self.samples[real_idx]
        # 获取该图像的所有关键点数据
        img_data = self.df[self.df['图像ID'] == image_id]
        
        # 获取图像文件名
        img_path = os.path.join(self.img_dir, self.filename_map[short_id])
        
        # 读取图像
        image = Image.open(img_path).convert('RGB')
        # 先调整到512x512（保持与原始处理一致）
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        # 然后调整到网络输入尺寸
        image = image.resize(self.IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # 应用数据增强（如果是训练集的增强样本）
        if is_augmented:
            image = self.augment(image)
        
        # 获取关键点坐标
        keypoints = []
        for point in ['CC1', 'CC2', 'CN1', 'CN2', 'N']:
            point_data = img_data[img_data['关键点'] == point].iloc[0]
            x = point_data['归一化X']
            y = point_data['归一化Y']
            keypoints.append([x, y])
        
        keypoints = np.array(keypoints, dtype=np.float32)
        
        # 生成热图目标
        target = self.generate_target(keypoints)
        
        # 转换图像
        if self.transform:
            image = self.transform(image)
        
        # 准备元数据（用于评估）
        img_width, img_height = self.IMAGE_SIZE[1], self.IMAGE_SIZE[0]  # W, H
        pix_pts = keypoints * np.array([img_width, img_height])  # 归一化 → 像素坐标
        
        meta = {
            'index': real_idx,
            'pts': torch.from_numpy(pix_pts).float(),  # 使用像素坐标
            'is_augmented': is_augmented
        }
        
        return image, target, meta

class HRNet(nn.Module):
    """简化版HRNet"""
    def __init__(self, num_joints):
        super(HRNet, self).__init__()
        
        # 主干网络 - 降采样到1/4
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 高分辨率分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 低分辨率分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fuse = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 输出头
        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_joints, kernel_size=1)
        )
    
    def forward(self, x):
        # 输入: 256x256
        x = self.conv1(x)  # 128x128
        x = self.conv2(x)  # 64x64
        
        # 分支处理
        x1 = self.branch1(x)  # 64x64
        x2 = self.branch2(x)  # 32x32
        
        # 上采样低分辨率特征
        x2_up = nn.functional.interpolate(
            x2, size=x1.shape[2:], mode='bilinear', align_corners=False
        )  # 32x32 -> 64x64
        
        # 特征融合
        x = torch.cat([x1, x2_up], dim=1)  # 64x64
        x = self.fuse(x)  # 64x64
        
        # 生成热图
        x = self.final_layer(x)  # 64x64
        
        return x  # 输出: 64x64

class DynamicWeightedLoss(nn.Module):
    """动态权重平衡的损失函数
    
    特点:
    1. 每个关键点通道独立计算BCEWithLogitsLoss
    2. 使用EMA维护每个通道的历史损失
    3. 动态调整权重，loss大的点权重增加
    4. 权重归一化确保训练稳定
    """
    def __init__(self, num_joints=5, momentum=0.9):
        super(DynamicWeightedLoss, self).__init__()
        self.num_joints = num_joints
        self.momentum = momentum  # EMA的动量因子
        
        # 初始化每个通道的损失函数
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # 初始化EMA记录器
        self.register_buffer('loss_ema', torch.ones(num_joints))
        self.register_buffer('weights', torch.ones(num_joints))
    
    def update_weights(self, losses):
        """更新每个通道的权重"""
        # 更新EMA
        self.loss_ema = self.momentum * self.loss_ema + (1 - self.momentum) * losses
        
        # 计算动态权重：总均值/每通道EMA的反比
        mean_loss = self.loss_ema.mean()
        weights = mean_loss / (self.loss_ema + 1e-8)  # 防止除零
        
        # 归一化权重，确保均值为1
        weights = weights * (self.num_joints / weights.sum())
        
        self.weights = weights
        return weights
    
    def forward(self, pred, target):
        """前向传播计算损失
        
        Args:
            pred: 预测热图 [B, K, H, W]
            target: 目标热图 [B, K, H, W]
        """
        batch_size = pred.size(0)
        
        # 对每个通道分别计算损失
        channel_losses = []
        for i in range(self.num_joints):
            # 将预测和目标展平
            pred_flat = pred[:, i].view(batch_size, -1)
            target_flat = target[:, i].view(batch_size, -1)
            
            # 计算该通道的BCE损失
            loss = self.criterion(pred_flat, target_flat).mean()
            channel_losses.append(loss)
        
        # 转换为tensor
        channel_losses = torch.stack(channel_losses)
        
        # 更新并获取动态权重
        weights = self.update_weights(channel_losses.detach())
        
        # 应用权重并求和
        weighted_loss = (weights * channel_losses).sum()
        
        return weighted_loss, channel_losses

def generate_target(target, point, sigma):
    """生成高斯热图
    Args:
        target: 目标热图 [H, W]
        point: 关键点坐标 [x, y] - x对应W, y对应H
        sigma: 高斯核大小
    """
    tmp_size = sigma * 3
    h, w = target.shape[:2]  # h对应height(y), w对应width(x)
    
    # 确保point的x对应width, y对应height
    mu_x, mu_y = int(point[0] + 0.5), int(point[1] + 0.5)
    
    # 计算热图范围 - 保持[x,y]顺序一致
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]  # [x,y]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]  # [x,y]
    
    # 如果关键点在图像外，跳过
    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return target
        
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0, y0 = size // 2, size // 2
    
    # 生成高斯分布
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    
    # 裁剪范围 - 严格保持x对应width, y对应height
    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]  # width方向
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]  # height方向
    img_x = max(0, ul[0]), min(br[0], w)  # width方向
    img_y = max(0, ul[1]), min(br[1], h)  # height方向
    
    # 确保target[h,w]和g[h,w]的对应关系正确
    target[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return target

def get_max_preds(heatmaps):
    """从热图中获取关键点预测位置"""
    batch_size = heatmaps.size(0)
    num_joints = heatmaps.size(1)
    width = heatmaps.size(3)  # W
    height = heatmaps.size(2)  # H
    
    heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
    idx = torch.argmax(heatmaps_reshaped, 2)
    maxvals = torch.amax(heatmaps_reshaped, 2)
    
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints))
    
    preds = torch.zeros((batch_size, num_joints, 2), device=heatmaps.device)
    
    # 确保x对应W, y对应H
    preds[:, :, 0] = (idx % width).float()  # x坐标(W)
    preds[:, :, 1] = (idx.div(width, rounding_mode='floor')).float()  # y坐标(H)
    
    pred_mask = maxvals > 0.0
    
    return preds, maxvals, pred_mask

def visualize_keypoints(image, keypoints, predictions=None, save_path=None, is_normalized=True, image_size=(256, 256)):
    """可视化关键点预测结果"""
    try:
        # 图像预处理
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
        
        # 如果是子图模式，不创建新的figure
        if plt.get_fignums() and plt.gcf().get_axes():
            ax = plt.gca()
        else:
            plt.figure(figsize=(8, 8))
            ax = plt.gca()
        
        ax.imshow(image)
        ax.axis('off')  # 关闭坐标轴显示
        
        # 定义关键点名称和颜色
        keypoint_names = ['CC1', 'CC2', 'CN1', 'CN2', 'N']
        colors = ['r', 'r', 'g', 'g', 'b']
        
        # 坐标转换
        if keypoints is not None:
            keypoints = keypoints.cpu().numpy() if isinstance(keypoints, torch.Tensor) else keypoints
            if is_normalized:
                keypoints = keypoints.copy() * np.array([image_size[1], image_size[0]])  # W,H
        
        if predictions is not None:
            predictions = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
            if is_normalized:
                predictions = predictions.copy() * np.array([image_size[1], image_size[0]])  # W,H
        
        # 绘制真实关键点
        if keypoints is not None:
            for i, (kp, name, color) in enumerate(zip(keypoints, keypoint_names, colors)):
                ax.plot(kp[0], kp[1], color + 'o', markersize=8, label=f'{name}_GT')
                ax.text(kp[0]+5, kp[1]+5, name, color=color, fontsize=10)
        
        # 绘制预测关键点
        if predictions is not None:
            for i, (pred, name, color) in enumerate(zip(predictions, keypoint_names, colors)):
                ax.plot(pred[0], pred[1], color + 'x', markersize=8, label=f'{name}_Pred')
                if keypoints is not None:
                    # 绘制从预测点到真实点的连线
                    ax.plot([pred[0], keypoints[i][0]], [pred[1], keypoints[i][1]], 
                            color + '--', alpha=0.5)
        
        # 添加图例
        if keypoints is not None or predictions is not None:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 保存或关闭
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            # 如果不是子图模式，关闭当前figure
            if not plt.gcf().get_axes() or len(plt.gcf().get_axes()) == 1:
                plt.close()
            
    except Exception as e:
        print(f"可视化过程中出错: {str(e)}")
        plt.close('all')  # 发生错误时关闭所有figure

def visualize_heatmap(image, target, pred, save_path):
    """可视化热图预测结果"""
    try:
        # 转换图像格式
        image = image.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        
        point_names = ['CC1', 'CC2', 'CN1', 'CN2', 'N']
        
        # 为每个关键点创建单独的图像
        for i in range(5):
            plt.figure(figsize=(15, 5))
            
            # 原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            # 目标热图
            plt.subplot(1, 3, 2)
            plt.imshow(image)
            target_heatmap = target[i].cpu().numpy()
            plt.imshow(target_heatmap, alpha=0.6, cmap='jet')
            plt.title(f'Target Heatmap - {point_names[i]}')
            plt.axis('off')
            
            # 预测热图
            plt.subplot(1, 3, 3)
            plt.imshow(image)
            pred_heatmap = pred[i].detach().cpu().numpy()
            plt.imshow(pred_heatmap, alpha=0.6, cmap='jet')
            plt.title(f'Predicted Heatmap - {point_names[i]}')
            plt.axis('off')
            
            plt.tight_layout()
            # 为每个关键点保存单独的图像
            point_save_path = save_path.replace('.png', f'_{point_names[i]}.png')
            plt.savefig(point_save_path, bbox_inches='tight', dpi=200)
            plt.close()
            
    except Exception as e:
        print(f"热图可视化过程中出错: {str(e)}")
        plt.close('all')  # 发生错误时关闭所有figure

def calc_metrics(output, target, meta):
    """计算评估指标"""
    pred, maxvals, _ = get_max_preds(output)
    target_coords = meta['pts']  # 使用像素坐标
    
    # 转换预测坐标到原始图像尺寸 (确保x对应W, y对应H)
    scale_factor = 256 / 64  # 从热图尺寸(64x64)到图像尺寸(256x256)的缩放因子
    pred = pred * scale_factor
    
    # 确保设备一致
    if pred.device != target_coords.device:
        target_coords = target_coords.to(pred.device)
    
    # 计算平均误差（像素）
    err = torch.norm(pred - target_coords, dim=2)
    err_mean = err.mean()
    
    return err_mean

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, vis_dir):
    model.train()
    losses = AverageMeter()
    metrics = AverageMeter()
    channel_losses = [AverageMeter() for _ in range(5)]  # 记录每个通道的损失
    
    os.makedirs(vis_dir, exist_ok=True)
    
    point_names = ['CC1', 'CC2', 'CN1', 'CN2', 'N']
    
    for i, (input, target, meta) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        
        # 前向传播
        output = model(input)
        loss, curr_channel_losses = criterion(output, target)
        
        # 计算评估指标
        err = calc_metrics(output.detach(), target, meta)
        metrics.update(err.item(), input.size(0))
        
        # 更新每个通道的损失
        for j, c_loss in enumerate(curr_channel_losses):
            channel_losses[j].update(c_loss.item(), input.size(0))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.update(loss.item(), input.size(0))
        
        # 打印训练信息
        if i % 10 == 0:
            # 构建每个关键点的损失信息
            channel_loss_msg = ' '.join([
                f'{name}: {meter.avg:.3f}({criterion.weights[j].item():.2f})'
                for j, (name, meter) in enumerate(zip(point_names, channel_losses))
            ])
            
            msg = f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t' \
                  f'Loss {losses.val:.5f} ({losses.avg:.5f})\t' \
                  f'Error {metrics.val:.2f}px ({metrics.avg:.2f}px)\n' \
                  f'Channel Losses: {channel_loss_msg}'
            logger.info(msg)
            
            # 可视化第一个样本的预测结果
            if i % 50 == 0:
                save_path = os.path.join(vis_dir, f'epoch_{epoch}_iter_{i}.png')
                pred, _, _ = get_max_preds(output.detach())
                pred = pred * 4  # 64 -> 256
                visualize_keypoints(
                    input[0],
                    meta['pts'][0],
                    pred[0],
                    save_path,
                    is_normalized=False,
                    image_size=(256, 256)
                )
    
    return losses.avg, metrics.avg, [meter.avg for meter in channel_losses]

def validate(model, val_loader, criterion, device, epoch, vis_dir):
    model.eval()
    losses = AverageMeter()
    metrics = AverageMeter()
    channel_losses = [AverageMeter() for _ in range(5)]
    
    os.makedirs(vis_dir, exist_ok=True)
    
    # 用于存储所有验证样本
    all_samples = []
    
    with torch.no_grad():
        for i, (input, target, meta) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)
            
            output = model(input)
            loss, curr_channel_losses = criterion(output, target)
            
            # 计算评估指标
            err = calc_metrics(output, target, meta)
            metrics.update(err.item(), input.size(0))
            
            # 更新每个通道的损失
            for j, c_loss in enumerate(curr_channel_losses):
                channel_losses[j].update(c_loss.item(), input.size(0))
            
            losses.update(loss.item(), input.size(0))
            
            # 存储当前batch的所有样本用于后续可视化
            pred, _, _ = get_max_preds(output)
            pred = pred * 4  # 64 -> 256
            for b in range(input.size(0)):
                all_samples.append({
                    'input': input[b].cpu(),
                    'pts': meta['pts'][b].cpu(),
                    'pred': pred[b].cpu()
                })
    
    # 随机选择4个样本进行可视化
    if len(all_samples) >= 4:
        selected_samples = random.sample(all_samples, 4)
        
        # 创建2x2的子图布局
        plt.figure(figsize=(16, 16))
        for idx, sample in enumerate(selected_samples):
            plt.subplot(2, 2, idx + 1)
            visualize_keypoints(
                sample['input'],
                sample['pts'],
                sample['pred'],
                None,  # 不单独保存每个子图
                is_normalized=False,
                image_size=(256, 256)
            )
        
        # 保存包含4个子图的整体图像
        save_path = os.path.join(vis_dir, f'val_epoch_{epoch}_samples.png')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()
    
    point_names = ['CC1', 'CC2', 'CN1', 'CN2', 'N']
    channel_loss_msg = ' '.join([
        f'{name}: {meter.avg:.3f}'
        for name, meter in zip(point_names, channel_losses)
    ])
    
    msg = f'Validation: Loss {losses.avg:.5f}\tError {metrics.avg:.2f}px\n' \
          f'Channel Losses: {channel_loss_msg}'
    logger.info(msg)
    
    return losses.avg, metrics.avg, [meter.avg for meter in channel_losses]

class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = KeypointDataset(
        csv_file='Second_model/第二模型标注结果_512px.csv',
        split_file='Second_model/数据集划分.csv',
        img_dir=r'D:\upm_model train\Second_model\处理后图片',
        is_train=True,
        transform=transform
    )
    
    val_dataset = KeypointDataset(
        csv_file='Second_model/第二模型标注结果_512px.csv',
        split_file='Second_model/数据集划分.csv',
        img_dir=r'D:\upm_model train\Second_model\处理后图片',
        is_train=False,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 创建模型
    model = HRNet(num_joints=KeypointDataset.NUM_JOINTS).to(device)
    
    # define loss function and optimizer
    criterion = DynamicWeightedLoss(num_joints=KeypointDataset.NUM_JOINTS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # initial learning rate
    
    # define learning rate scheduler
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            # first 40 epochs, use cosine annealing
            optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,  # restart every 10 epochs
                T_mult=2,  # double the period after each restart
                eta_min=1e-5  # minimum learning rate
            ),
            # second stage: last 40 epochs, use multi-step decay for fine-tuning
            optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[40, 60],  # reduce learning rate at 40 and 60 epochs
                gamma=0.1  # reduce to 0.1 times
            )
        ],
        milestones=[40]  # switch to second scheduler at 40 epochs
    )
    
    # 创建保存目录
    save_dir = Path('checkpoints')
    save_dir.mkdir(exist_ok=True)
    
    # 创建可视化目录
    vis_dir = Path('visualizations')
    vis_dir.mkdir(exist_ok=True)
    
    # 用于绘制损失曲线
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    channel_train_losses = [[] for _ in range(5)]  # 每个通道的训练损失历史
    channel_val_losses = [[] for _ in range(5)]    # 每个通道的验证损失历史
    
    # 训练循环
    num_epochs = 300  # 总共300轮：200轮基础训练 + 100轮精细微调
    best_loss = float('inf')
    best_error = float('inf')
    
    for epoch in range(num_epochs):
        logger.info(f'\nEpoch {epoch + 1}/{num_epochs}')
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'当前学习率: {current_lr:.2e}')
        
        # 训练一个epoch
        train_loss, train_err, train_channel_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            vis_dir / 'train'
        )
        
        # 验证
        val_loss, val_err, val_channel_losses = validate(
            model, val_loader, criterion, device, epoch,
            vis_dir / 'val'
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录损失和指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics.append(train_err)
        val_metrics.append(val_err)
        
        # 记录每个通道的损失
        for i in range(5):
            channel_train_losses[i].append(train_channel_losses[i])
            channel_val_losses[i].append(val_channel_losses[i])
        
        # 绘制损失曲线
        plt.figure(figsize=(15, 10))
        
        # 总体损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.axvline(x=200, color='r', linestyle='--', label='Phase Change')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Overall Loss')
        plt.legend()
        plt.grid(True)
        
        # 误差曲线
        plt.subplot(2, 2, 2)
        plt.plot(train_metrics, label='Train Error (px)')
        plt.plot(val_metrics, label='Val Error (px)')
        plt.axvline(x=200, color='r', linestyle='--', label='Phase Change')
        plt.xlabel('Epoch')
        plt.ylabel('Average Error (pixels)')
        plt.title('Prediction Error')
        plt.legend()
        plt.grid(True)
        
        # 每个通道的训练损失
        plt.subplot(2, 2, 3)
        point_names = ['CC1', 'CC2', 'CN1', 'CN2', 'N']
        for i, name in enumerate(point_names):
            plt.plot(channel_train_losses[i], label=name)
        plt.axvline(x=200, color='r', linestyle='--', label='Phase Change')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Channel-wise Training Loss')
        plt.legend()
        plt.grid(True)
        
        # 每个通道的验证损失
        plt.subplot(2, 2, 4)
        for i, name in enumerate(point_names):
            plt.plot(channel_val_losses[i], label=name)
        plt.axvline(x=200, color='r', linestyle='--', label='Phase Change')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Channel-wise Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'training_progress.png')
        plt.close()
        
        logger.info(f'Train Loss: {train_loss:.4f}, Error: {train_err:.2f}px')
        logger.info(f'Val Loss: {val_loss:.4f}, Error: {val_err:.2f}px')
        
        # 保存最佳模型（迭代式保存）
        if val_loss < best_loss:
            best_loss = val_loss
            best_error = val_err
            
            # 生成包含时间戳的模型文件名
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            model_filename = f'best_model_epoch_{epoch+1}_loss_{val_loss:.4f}_err_{val_err:.2f}_{timestamp}.pth'
            
            # 保存模型
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'criterion_state': criterion.state_dict(),
                'best_loss': best_loss,
                'best_error': best_error,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'channel_train_losses': channel_train_losses,
                'channel_val_losses': channel_val_losses
            }, save_dir / model_filename)
            
            logger.info(f'保存新的最佳模型: {model_filename}')
        
        # 每50个epoch保存一个常规检查点
        if (epoch + 1) % 50 == 0:
            checkpoint_filename = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'criterion_state': criterion.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'channel_train_losses': channel_train_losses,
                'channel_val_losses': channel_val_losses
            }, save_dir / checkpoint_filename)
            
            logger.info(f'保存检查点: {checkpoint_filename}')

if __name__ == '__main__':
    main() 
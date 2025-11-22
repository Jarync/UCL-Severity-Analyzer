# ------------------------------------------------------------------
#  Train / Validate / Inference helpers
#  修改：统一 device；去掉所有 .cuda()
# ------------------------------------------------------------------
from __future__ import absolute_import, division, print_function

import time, logging, os
import torch, numpy as np
from tqdm import tqdm
from .evaluation import get_preds
from ..datasets import visualize_keypoints

logger = logging.getLogger(__name__)


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


# 简单按比例把 heatmap 坐标放大到原图像素
def simple_decode_preds(heatmaps, image_size, heatmap_size):
    """
    将热图上的坐标直接按比例转换到原图像素坐标
    
    参数:
        heatmaps: 热图张量 (B, J, H, W) 
        image_size: 原图尺寸 [width, height]
        heatmap_size: 热图尺寸 [width, height]
    
    返回:
        预测坐标 (B, J, 2) - 原图像素坐标
    """
    coords = get_preds(heatmaps)  # (B, J, 2) - 1-indexed坐标
    coords = coords - 1  # 转为0-indexed
    
    # 计算缩放比例
    stride_w = image_size[0] / heatmap_size[0]
    stride_h = image_size[1] / heatmap_size[1]
    
    # 缩放到原图尺寸
    coords = coords.clone()
    coords[:, :, 0] *= stride_w  # x坐标
    coords[:, :, 1] *= stride_h  # y坐标
    
    return coords


# ------------------------------------------------------------------
def train(config, train_loader, model, criterion, optimizer,
          epoch, writer_dict, device):

    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    model.train()

    nme_sum, nme_cnt = 0.0, 0
    end = time.time()

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for i, (inp, target, meta) in enumerate(progress_bar):
        data_time.update(time.time() - end)

        inp    = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(inp)
        loss   = criterion(output, target)
        
        # 调试代码：检查输出和目标尺寸是否匹配
        if i == 0:
            print(f'DEBUG - output/target size: {output.shape}, {target.shape}')
            assert output.shape == target.shape, \
                f'Size mismatch! output={output.shape} target={target.shape}'

        # -------- NME --------
        score_map = output.detach().cpu()
        preds = simple_decode_preds(score_map, config.MODEL.IMAGE_SIZE, config.MODEL.HEATMAP_SIZE)
        
        # 计算NME (Normalized Mean Error)
        batch_size = preds.size(0)
        nme_batch_sum = 0
        for n in range(batch_size):
            pred = preds[n]  # (J, 2)
            gt = meta['pts'][n]  # (J, 2)
            
            if gt.sum() > 0:  # 如果有标注点
                # 计算欧氏距离
                distances = torch.sqrt(((pred - gt) ** 2).sum(dim=1))
                # 使用眼距作为归一化因子
                if gt[0].sum() > 0 and gt[1].sum() > 0:  # 如果眼睛标注存在
                    eye_distance = torch.sqrt(((gt[0] - gt[1]) ** 2).sum())
                    eye_distance = max(eye_distance, 1.0)  # 避免除零
                else:
                    eye_distance = torch.tensor(100.0)  # 默认眼距
                
                # 计算NME
                nme = (distances.mean() / eye_distance).item()
                nme_batch_sum += nme
        
        nme_sum += nme_batch_sum
        nme_cnt += batch_size

        # -------- optimize --------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))
        batch_time.update(time.time() - end)
        
        # 更新进度条信息
        progress_bar.set_postfix({
            'loss': f'{losses.val:.5f}({losses.avg:.5f})', 
            'time': f'{batch_time.val:.3f}({batch_time.avg:.3f})'
        })

        if i % config.PRINT_FREQ == 0:
            speed = inp.size(0) / max(batch_time.val, 1e-5)
            logger.info(
                f'Epoch:[{epoch}][{i}/{len(train_loader)}]  '
                f'Time {batch_time.val:.3f}({batch_time.avg:.3f})  '
                f'Speed {speed:.1f}/s  '
                f'Data {data_time.val:.3f}({data_time.avg:.3f})  '
                f'Loss {losses.val:.5f}({losses.avg:.5f})'
            )
            if writer_dict:
                w = writer_dict['writer']
                gs = writer_dict['train_global_steps']
                w.add_scalar('train_loss', losses.val, gs)
                writer_dict['train_global_steps'] = gs + 1

        end = time.time()

    nme = nme_sum / max(nme_cnt, 1)
    logger.info(f'Train Epoch {epoch}  time:{batch_time.avg:.3f}  '
                f'loss:{losses.avg:.4f}  nme:{nme:.4f}')


# ------------------------------------------------------------------
def validate(config, val_loader, model, criterion,
             epoch, writer_dict, device):

    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()

    nme_sum, nme_cnt = 0.0, 0
    fail_008 = fail_010 = 0
    preds_all = torch.zeros((len(val_loader.dataset),
                             config.MODEL.NUM_JOINTS, 2))

    # 创建可视化结果保存目录
    vis_dir = os.path.join(config.OUTPUT_DIR, config.DATASET.DATASET, 
                         os.path.basename(config.MODEL.PRETRAINED).split('.')[0], 'vis_results')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 每个epoch创建子目录
    epoch_vis_dir = os.path.join(vis_dir, f'epoch_{epoch}')
    os.makedirs(epoch_vis_dir, exist_ok=True)
    
    # 保存一些样本的原始图像和预测结果，用于可视化
    samples_to_visualize = []

    end = time.time()
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for i, (inp, target, meta) in enumerate(progress_bar):
            data_time.update(time.time() - end)

            inp    = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(inp)
            loss   = criterion(output, target)

            score_map = output.detach().cpu()
            preds = simple_decode_preds(score_map, config.MODEL.IMAGE_SIZE, config.MODEL.HEATMAP_SIZE)

            # 计算NME
            batch_size = preds.size(0)
            for n in range(batch_size):
                pred = preds[n]  # (J, 2)
                gt = meta['pts'][n]  # (J, 2)
                
                if gt.sum() > 0:  # 如果有标注点
                    # 计算欧氏距离
                    distances = torch.sqrt(((pred - gt) ** 2).sum(dim=1))
                    # 使用眼距作为归一化因子
                    if gt[0].sum() > 0 and gt[1].sum() > 0:  # 如果眼睛标注存在
                        eye_distance = torch.sqrt(((gt[0] - gt[1]) ** 2).sum())
                        eye_distance = max(eye_distance, 1.0)  # 避免除零
                    else:
                        eye_distance = torch.tensor(100.0)  # 默认眼距
                    
                    # 计算NME
                    nme = (distances.mean() / eye_distance).item()
                    nme_batch = torch.tensor([nme])
                    
                    # 计算失败率
                    fail_008 += (nme_batch > 0.08).sum().item()
                    fail_010 += (nme_batch > 0.10).sum().item()
                    nme_sum += nme
                    nme_cnt += 1
                
                # 保存预测结果
                preds_all[meta['index'][n]] = preds[n]
                
                # 收集一些样本进行可视化（每个epoch保存前10个样本）
                if len(samples_to_visualize) < 10:
                    samples_to_visualize.append({
                        'image': inp[n].cpu(),
                        'prediction': preds[n],
                        'target': meta['pts'][n],
                        'index': meta['index'][n].item()
                    })

            losses.update(loss.item(), inp.size(0))
            batch_time.update(time.time() - end)
            
            # 更新进度条信息
            progress_bar.set_postfix({
                'loss': f'{losses.val:.5f}({losses.avg:.5f})', 
                'time': f'{batch_time.val:.3f}({batch_time.avg:.3f})'
            })
            
            end = time.time()
    
    # 生成可视化结果
    if epoch % 5 == 0:  # 每5个epoch生成一次可视化结果
        logger.info(f'Saving visualization results to {epoch_vis_dir}')
        for i, sample in enumerate(samples_to_visualize):
            img = sample['image']
            pred = sample['prediction']
            target = sample['target']
            idx = sample['index']
            
            # 生成可视化并保存
            save_path = os.path.join(epoch_vis_dir, f'sample_{idx}_pred.png')
            visualize_keypoints(
                image=img, 
                keypoints=target,  # 真实关键点 
                predictions=pred,  # 预测关键点
                save_path=save_path,
                is_normalized=False,  # 这些坐标已经是像素坐标了
                image_size=tuple(config.MODEL.IMAGE_SIZE)
            )

    nme   = nme_sum / max(nme_cnt, 1)
    f008  = fail_008 / max(nme_cnt, 1)
    f010  = fail_010 / max(nme_cnt, 1)

    logger.info(
        f'Val  Epoch {epoch}  time:{batch_time.avg:.3f}  '
        f'loss:{losses.avg:.4f}  nme:{nme:.4f}  '
        f'[>0.08]:{f008:.4f}  [>0.10]:{f010:.4f}'
    )

    if writer_dict:
        w  = writer_dict['writer']
        gs = writer_dict['valid_global_steps']
        w.add_scalar('valid_loss', losses.avg, gs)
        w.add_scalar('valid_nme',  nme,         gs)
        writer_dict['valid_global_steps'] = gs + 1

    return nme, preds_all


# ------------------------------------------------------------------
def inference(config, data_loader, model, device):
    """Only forward & metric — no loss/grad; returns nme and preds tensor"""
    batch_time, data_time = AverageMeter(), AverageMeter()
    nme_sum, nme_cnt = 0.0, 0
    fail_008 = fail_010 = 0

    preds_all = torch.zeros((len(data_loader.dataset),
                             config.MODEL.NUM_JOINTS, 2))
    model.eval()
    end = time.time()
    
    # 创建预测可视化结果保存目录
    vis_dir = os.path.join(config.OUTPUT_DIR, config.DATASET.DATASET, 
                          os.path.basename(config.MODEL.PRETRAINED).split('.')[0], 'inference_results')
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Inference')
        for i, (inp, _, meta) in enumerate(progress_bar):
            data_time.update(time.time() - end)
            inp = inp.to(device, non_blocking=True)
            output = model(inp)

            score_map = output.detach().cpu()
            preds = simple_decode_preds(score_map, config.MODEL.IMAGE_SIZE, config.MODEL.HEATMAP_SIZE)

            # 计算NME
            batch_size = preds.size(0)
            for n in range(batch_size):
                pred = preds[n]  # (J, 2)
                gt = meta['pts'][n]  # (J, 2)
                
                if gt.sum() > 0:  # 如果有标注点
                    # 计算欧氏距离
                    distances = torch.sqrt(((pred - gt) ** 2).sum(dim=1))
                    # 使用眼距作为归一化因子
                    if gt[0].sum() > 0 and gt[1].sum() > 0:  # 如果眼睛标注存在
                        eye_distance = torch.sqrt(((gt[0] - gt[1]) ** 2).sum())
                        eye_distance = max(eye_distance, 1.0)  # 避免除零
                    else:
                        eye_distance = torch.tensor(100.0)  # 默认眼距
                    
                    # 计算NME
                    nme = (distances.mean() / eye_distance).item()
                    nme_batch = torch.tensor([nme])
                    
                    # 计算失败率
                    fail_008 += (nme_batch > 0.08).sum().item()
                    fail_010 += (nme_batch > 0.10).sum().item()
                    nme_sum += nme
                    nme_cnt += 1
                
                idx = meta['index'][n].item()
                preds_all[meta['index'][n]] = preds[n]
                
                # 保存预测结果可视化
                if i * inp.size(0) + n < 50:  # 只保存前50个样本的可视化结果
                    save_path = os.path.join(vis_dir, f'sample_{idx}_pred.png')
                    visualize_keypoints(
                        image=inp[n].cpu(), 
                        keypoints=meta['pts'][n],  # 真实关键点（如果有）
                        predictions=preds[n],      # 预测关键点
                        save_path=save_path,
                        is_normalized=False,       # 这些坐标已经是像素坐标了
                        image_size=tuple(config.MODEL.IMAGE_SIZE)
                    )

            batch_time.update(time.time() - end)
            
            # 更新进度条信息
            progress_bar.set_postfix({
                'time': f'{batch_time.val:.3f}({batch_time.avg:.3f})'
            })
            
            end = time.time()

    nme = nme_sum / max(nme_cnt, 1)
    f008 = fail_008 / max(nme_cnt, 1)
    f010 = fail_010 / max(nme_cnt, 1)
    
    logger.info(
        f'Inference  time:{batch_time.avg:.3f}  nme:{nme:.4f}  '
        f'[>0.08]:{f008:.4f}  [>0.10]:{f010:.4f}'
    )
    return nme, preds_all

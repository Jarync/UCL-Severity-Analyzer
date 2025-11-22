# ------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified by <your-name> for universal NME (any landmark number)
# ------------------------------------------------------------------------

import math
import numpy as np
import torch

from ..utils.transforms import transform_preds


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------
def get_preds(scores: torch.Tensor) -> torch.Tensor:
    """
    从 heat-map 获取最大响应坐标 (1-based)，返回 shape (B, J, 2)
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim (B,C,H,W)'
    # 找最大值索引
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)
    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx    = idx.view(scores.size(0), scores.size(1), 1) + 1  # 1-based

    preds = idx.repeat(1, 1, 2).float()
    preds[..., 0] = (preds[..., 0] - 1) % scores.size(3) + 1    # x
    preds[..., 1] = torch.floor((preds[..., 1] - 1) / scores.size(3)) + 1  # y

    mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= mask
    return preds


# ----------------------------------------------------------------------
# 通用 NME 计算
# ----------------------------------------------------------------------
def _diag_len(w: float, h: float) -> float:
    return math.hypot(w, h)


def compute_nme(preds: torch.Tensor, meta: dict) -> np.ndarray:
    """
    Normalized Mean Error  
    * 支持任意关键点数量
    * 优先用第 0,1 两点距离归一化；否则用对角线
    """
    preds   = preds.cpu().numpy()                # (B,J,2)  像素坐标
    targets = meta['pts'].cpu().numpy()          # (B,J,2)  归一化 or 像素都行
    B, J, _ = preds.shape
    rmse    = np.zeros(B, dtype=np.float32)

    for i in range(B):
        # 可见点 mask（target x 或 y < 0 视为不可见）
        vis = (targets[i,:,0] >= 0) & (targets[i,:,1] >= 0)
        if not np.any(vis):
            continue

        diff = np.linalg.norm(preds[i,vis] - targets[i,vis], axis=1)
        mean_err = diff.mean()

        # ---------------- 归一化因子 ----------------
        norm = None
        if J >= 2 and vis[0] and vis[1]:
            norm = np.linalg.norm(targets[i,0] - targets[i,1])  # E1-E2
        # 有 box_size 字段时（AFLW）可用
        if norm is None or norm < 1e-6:
            if 'box_size' in meta:
                norm = float(meta['box_size'][i])
        # 兜底：用对角线（坐标系统单位 1×1）
        if norm is None or norm < 1e-6:
            norm = _diag_len(1.0, 1.0)
            
        # 确保归一化因子不会太小，防止NME值异常大
        norm = max(norm, 1.0)

        rmse[i] = mean_err / norm

    return rmse


# ----------------------------------------------------------------------
# heat-map → 真实坐标解码，保持原实现
# ----------------------------------------------------------------------
def decode_preds(output: torch.Tensor,
                 center: torch.Tensor,
                 scale: torch.Tensor,
                 res: tuple[int, int]) -> torch.Tensor:
    """
    将网络输出 heat-map 解码回图像坐标
    """
    coords = get_preds(output)       # (B,J,2)

    coords = coords.cpu()
    # 二次插值微调
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n,p,0]))
            py = int(math.floor(coords[n,p,1]))
            if 1 < px < res[0] and 1 < py < res[1]:
                diff = torch.tensor([hm[py-1, px] - hm[py-1, px-2],
                                    hm[py,   px-1] - hm[py-2, px-1]])
                coords[n,p] += diff.sign() * 0.25
    coords += 0.5
    preds = coords.clone()

    # 反向映射到原图
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, *preds.size())

    return preds

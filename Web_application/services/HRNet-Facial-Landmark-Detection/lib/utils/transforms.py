# ------------------------------------------------------------------------------
#  Custom HRNet utilities (CV-only version)
#  – Replaces original Microsoft implementation that relied on deprecated
#    scipy.misc.imresize / imrotate.
# ------------------------------------------------------------------------------
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch

__all__ = [
    "MATCHED_PARTS",
    "fliplr_joints",
    "get_affine_transform",
    "crop_v2",
    "get_transform",
    "transform_pixel",
    "transform_preds",
    "crop",
    "generate_target",
]

# --------------------------------------------------------------------------------------
#   Symmetric landmark indices for horizontal flip
#   • 300W / AFLW / COFW use 1-based indexing in the original repo – keep它们兼容
#   • WFLW 与我们自定义的 CLEFT 均为 0-based
# --------------------------------------------------------------------------------------
MATCHED_PARTS: dict[str, Sequence[Tuple[int, int]]] = {
    "300W": ([1, 17], [2, 16], [3, 15], [4, 14], [5, 13], [6, 12], [7, 11], [8, 10],
              [18, 27], [19, 26], [20, 25], [21, 24], [22, 23],
              [32, 36], [33, 35],
              [37, 46], [38, 45], [39, 44], [40, 43], [41, 48], [42, 47],
              [49, 55], [50, 54], [51, 53], [62, 64], [61, 65], [68, 66],
              [59, 57], [60, 56]),

    "AFLW": ([1, 6], [2, 5], [3, 4],
              [7, 12], [8, 11], [9, 10],
              [13, 15],
              [16, 18]),

    "COFW": ([1, 2], [5, 7], [3, 4], [6, 8], [9, 10], [11, 12], [13, 15],
              [17, 18], [14, 16], [19, 20], [23, 24]),

    "WFLW": ([0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26], [7, 25],
              [8, 24], [9, 23], [10, 22], [11, 21], [12, 20], [13, 19], [14, 18],
              [15, 17],
              [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49],
              [40, 48], [41, 47],
              [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74],
              [67, 73],
              [55, 59], [56, 58],
              [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
              [88, 92], [89, 91], [95, 93], [96, 97]),

    # 6-landmark Cleft-lip: (E1-E2, I1-I2, Nl-Nr) —— 0-based indices
    "CLEFT": ([0, 1], [2, 3], [4, 5]),
}


# =========================================================
#  Core helpers
# =========================================================

def fliplr_joints(pts: np.ndarray, width: int | float, *, dataset: str) -> np.ndarray:
    """水平翻转关键点数组

    Parameters
    ----------
    pts : (N,2) ndarray – 关键点 (x,y)
    width : int/float – 原图宽,
    dataset : str – key of MATCHED_PARTS
    """
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must be (N,2) array")

    # flip x coordinate
    pts[:, 0] = width - pts[:, 0]

    pairs = MATCHED_PARTS.get(dataset.upper(), ())
    zero_based = dataset.upper() in ("WFLW", "CLEFT")

    for a, b in pairs:
        if zero_based:
            ia, ib = a, b
        else:          # original datasets是 1-based
            ia, ib = a - 1, b - 1
        pts[[ia, ib]] = pts[[ib, ia]]
    return pts


# ---------------------------------------------------------
#  仿射矩阵 / 裁剪
# ---------------------------------------------------------

def _get_dir(src_pt: Sequence[float], rot_rad: float) -> np.ndarray:
    sn, cs = math.sin(rot_rad), math.cos(rot_rad)
    return np.array([src_pt[0] * cs - src_pt[1] * sn,
                     src_pt[0] * sn + src_pt[1] * cs], dtype=np.float32)


def get_affine_transform(center: np.ndarray | torch.Tensor,
                         scale: float | Sequence[float] | np.ndarray | torch.Tensor,
                         rot: float,
                         output_size: Tuple[int, int],
                         shift: np.ndarray | Sequence[float] = (0.0, 0.0),
                         *, inv: bool = False) -> np.ndarray:
    """完全沿用原 repo 逻辑, 只是改用 numpy."""
    center = np.asarray(center, dtype=np.float32)
    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)
    else:
        scale = np.asarray(scale, dtype=np.float32)

    scale_tmp = scale * 200.0
    src_w, src_h = scale_tmp[0], scale_tmp[1]
    dst_w, dst_h = float(output_size[0]), float(output_size[1])

    rot_rad = math.radians(rot)
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    # third point for perspective
    def _get_3rd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return b + np.array([-1, 1], dtype=np.float32) * (a - b)[::-1]

    src[2, :] = _get_3rd(src[0, :], src[1, :])
    dst[2, :] = _get_3rd(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(dst.astype(np.float32), src.astype(np.float32))
    else:
        trans = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))
    return trans  # 2×3


def crop_v2(img: np.ndarray,
            center: np.ndarray | torch.Tensor,
            scale: float | Sequence[float] | np.ndarray | torch.Tensor,
            output_size: Tuple[int, int],
            rot: float = 0.0) -> np.ndarray:
    trans = get_affine_transform(center, scale, rot, output_size)
    return cv2.warpAffine(img, trans, dsize=output_size, flags=cv2.INTER_LINEAR)


# ---------------------------------------------------------
#   Pixel/coord transforms (for decode/encode)
# ---------------------------------------------------------

def get_transform(center, scale, output_size, rot=0.0):
    # deprecated in internal code path – keep for compat
    return get_affine_transform(center, scale, rot, output_size, inv=False)


def transform_pixel(pt, center, scale, output_size, invert=0, rot=0.0):
    trans = get_affine_transform(center, scale, rot, output_size, inv=bool(invert))
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
    new_pt = trans @ new_pt
    return new_pt[:2]


def transform_preds(coords: torch.Tensor, center, scale, output_size):
    for p in range(coords.size(0)):
        coords[p, 0:2] = torch.from_numpy(
            transform_pixel(coords[p, 0:2].cpu().numpy(), center, scale, output_size, invert=1)
        )
    return coords


# ---------------------------------------------------------
#   Simple crop used in dataset augmentation (no SciPy!)
# ---------------------------------------------------------

def crop(img: np.ndarray,
         center: np.ndarray | torch.Tensor,
         scale: float | Sequence[float],
         output_size: Tuple[int, int],
         rot: float = 0.0) -> np.ndarray:
    return crop_v2(img, center, scale, output_size, rot)


# ---------------------------------------------------------
#  Heat-map generation (same as original)
# ---------------------------------------------------------

def generate_target(heatmap: np.ndarray,
                    pt: Sequence[float],
                    sigma: float,
                    *, label_type: str = "Gaussian") -> np.ndarray:
    tmp_size = sigma * 3
    mu_x, mu_y = int(pt[0] + 0.5), int(pt[1] + 0.5)

    w, h = heatmap.shape[1], heatmap.shape[0]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

    # skip if outside
    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return heatmap

    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    if label_type == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:  # Laplace
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]

    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)

    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return heatmap

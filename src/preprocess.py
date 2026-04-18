"""
图像预处理增强函数库 (Image Preprocessing Toolkit)

本模块封装了一组对扫描件二维码识别极为有效的图像增强算子, 是流水线
(pipeline.py / qr_repair.py) 的底层依赖。

主要算子:
  - crop_roi:           裁剪右上角 ROI 区域 (针对单据二维码的常见位置)
  - add_white_padding:  添加白色边框, 恢复二维码的 Quiet Zone
  - apply_clahe:        LAB 空间 L 通道的 CLAHE 局部对比度增强
  - sharpen:            3x3 拉普拉斯锐化
  - gamma_correct:      Gamma 校正 (gamma<1 提亮暗部, 增强对比度)
  - binarize_otsu:      Otsu 全局二值化
  - denoise_then_binarize: 高斯去噪 + Otsu (针对噪声扫描件)
  - rescale:            等比例缩放, 过小图像返回 None

实测经验:
  对于灰底单据扫描件, "gamma_correct(0.6) + apply_clahe" 是命中率最高的组合。
"""

from numpy import array as np_array, float32 as np_float32
from cv2 import (
    BORDER_CONSTANT,
    COLOR_BGR2GRAY,
    COLOR_BGR2LAB,
    COLOR_GRAY2BGR,
    COLOR_LAB2BGR,
    GaussianBlur,
    INTER_AREA,
    INTER_CUBIC,
    LUT,
    THRESH_BINARY,
    THRESH_OTSU,
    copyMakeBorder,
    createCLAHE,
    cvtColor,
    filter2D,
    merge as cv_merge,
    resize as cv_resize,
    split as cv_split,
    threshold,
)


# ── ROI 裁剪 ─────────────────────────────────────────────


def crop_roi(img, w_ratio: float, h_ratio: float):
    """裁剪图像的右上角 ROI (Region of Interest) 区域。

    本批单据的二维码统一位于右上角, 故只取该区域的子图,
    既能减小后续处理的计算量, 又能去除大面积无关背景对识别的干扰。

    Args:
        img:     输入 BGR 图像 (numpy ndarray)
        w_ratio: 宽度比例 (0~1), 表示从右边界向左裁剪的宽度占比
        h_ratio: 高度比例 (0~1), 表示从上边界向下裁剪的高度占比

    Returns:
        裁剪后的子图。例如 w_ratio=0.3, h_ratio=0.18 表示截取
        右上角 30% 宽 × 18% 高 的矩形区域。
    """
    h, w = img.shape[:2]
    x_start = max(0, int(w * (1 - w_ratio)))
    y_end = int(h * h_ratio)
    return img[0:y_end, x_start:w]


# ── 边框填充 ─────────────────────────────────────────────


def add_white_padding(img, pad: int = 80):
    """在图像四周添加固定厚度的白色边框, 用于恢复二维码的静区 (Quiet Zone)。

    QR 标准要求二维码周围至少留出 4 个模块宽度的纯色 (通常白色) 静区,
    扫描件中常因裁切丢失静区, 导致 WeChatQRCode 无法定位 Finder Pattern。
    本函数通过添加白边人工恢复静区, 是边缘裁切类失败样本的关键解药。

    Args:
        img: 输入 BGR 图像
        pad: 四个方向各添加的像素数 (典型值 80~200)

    Returns:
        添加边框后的图像 (尺寸 = 原尺寸 + 2*pad)
    """
    return copyMakeBorder(
        img, pad, pad, pad, pad, BORDER_CONSTANT, value=(255, 255, 255)
    )


# ── 对比度增强 ───────────────────────────────────────────


def apply_clahe(img, clip: float = 3.0, grid: tuple = (8, 8)):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) 局部对比度增强。

    将图像转换到 LAB 颜色空间, 仅对亮度通道 (L) 做自适应直方图均衡,
    避免了 BGR 直接均衡造成的颜色失真。对比度提升的同时保留色彩。

    Args:
        img:  输入 BGR 图像
        clip: 对比度限制阈值, 越大对比度提升越明显, 但也越容易放大噪声
        grid: 局部均衡的 tile 尺寸 (8x8 是默认推荐值)

    Returns:
        对比度增强后的 BGR 图像
    """
    lab = cvtColor(img, COLOR_BGR2LAB)
    l, a, b = cv_split(lab)
    clahe = createCLAHE(clipLimit=clip, tileGridSize=grid)
    l = clahe.apply(l)
    return cvtColor(cv_merge([l, a, b]), COLOR_LAB2BGR)


# ── 锐化 ─────────────────────────────────────────────────


def sharpen(img):
    """3x3 拉普拉斯锐化卷积核, 强化边缘细节。

    对模糊扫描件的边界恢复极为有效, 能让 WeChatQRCode 检测器
    更容易识别出 QR 码的模块边界。
    """
    kernel = np_array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np_float32
    )
    return filter2D(img, -1, kernel)


# ── Gamma 校正 ───────────────────────────────────────────


def gamma_correct(img, gamma: float = 0.6):
    """Gamma 校正 (幂函数变换)。

    输出像素 = 255 * (输入像素/255)^(1/gamma)

    - gamma < 1: 提亮暗部, 压暗亮部, 整体看起来更亮; 对低对比度灰底图最有效
    - gamma = 1: 不变
    - gamma > 1: 压暗暗部, 提亮亮部 (基本不用于本项目)

    实测:
      - gamma=0.6 是大多数扫描件的甜点值
      - gamma=0.3 用于极端低对比度场景 (灰底灰码)

    使用 LUT (Look-Up Table) 加速, 避免逐像素幂运算。
    """
    inv_gamma = 1.0 / gamma
    table = np_array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return LUT(img, table)


# ── 二值化 ───────────────────────────────────────────────


def binarize_otsu(img):
    """Otsu 自动阈值二值化, 输出仍为 BGR 三通道格式 (兼容下游函数)。

    Otsu 算法自动寻找使类间方差最大的阈值, 适合双峰分布的图像
    (二维码本身就是典型的双峰: 黑模块 + 白模块)。
    """
    gray = (
        cvtColor(img, COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    )
    _, bw = threshold(gray, 0, 255, THRESH_BINARY + THRESH_OTSU)
    return cvtColor(bw, COLOR_GRAY2BGR)


def denoise_then_binarize(img):
    """先高斯去噪再 Otsu 二值化, 抑制椒盐噪声对阈值估计的干扰。

    针对低质量扫描件 (有彩色噪点、压缩噪声) 比纯 Otsu 更稳。
    """
    gray = (
        cvtColor(img, COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    )
    blurred = GaussianBlur(gray, (3, 3), 0)
    _, bw = threshold(blurred, 0, 255, THRESH_BINARY + THRESH_OTSU)
    return cvtColor(bw, COLOR_GRAY2BGR)


# ── 缩放 ─────────────────────────────────────────────────


def rescale(img, scale: float):
    """等比例缩放图像。

    缩放方向选择最佳插值算法:
      - 放大 (scale > 1): INTER_CUBIC (双三次, 边缘平滑)
      - 缩小 (scale < 1): INTER_AREA  (区域采样, 抗锯齿)

    安全保护: 缩放后任一维度小于 50 像素则返回 None,
    避免给检测器送入过小的图像导致崩溃或无意义的耗时。

    Args:
        img:   输入图像
        scale: 缩放因子, 1.0 表示不缩放 (此时直接返回原图引用)

    Returns:
        缩放后的图像; 尺寸过小时返回 None。
    """
    if scale == 1.0:
        return img
    new_w = int(img.shape[1] * scale)
    new_h = int(img.shape[0] * scale)
    if new_w < 50 or new_h < 50:
        return None
    interp = INTER_CUBIC if scale > 1.0 else INTER_AREA
    return cv_resize(img, (new_w, new_h), interpolation=interp)

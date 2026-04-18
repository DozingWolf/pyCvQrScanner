"""
多层级二维码识别流水线 (Multi-Stage QR Recognition Pipeline)

架构设计:
  Layer 0: 原图直接识别 (最快, ~50ms)
  Layer 1: ROI 裁剪 + 白边填充 + Gamma/CLAHE 增强 + 多尺度缩放
  Layer 2: 全图增强 (CLAHE + 锐化) + 多尺度
  Layer 3: CTF 风格定位角修复 (由 qr_repair 模块提供)

短路求值: 任一层级识别成功即立即返回, 不再尝试后续层级。
超时熔断: 单张图片总处理时间硬限制 20 秒。
"""

from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from os.path import isfile
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple

from numpy import array as np_array, ndarray
from loguru import logger

# OpenCV — 显式导入用到的函数和常量, 替代裸 `import cv2`
from cv2 import (
    ADAPTIVE_THRESH_GAUSSIAN_C,
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
    addWeighted,
    adaptiveThreshold,
    copyMakeBorder,
    createCLAHE,
    cvtColor,
    imread,
    merge as cv_merge,
    resize as cv_resize,
    split as cv_split,
)
# WeChatQRCode 模型类 — opencv-contrib 子模块, 必须分开导入
from cv2.wechat_qrcode import WeChatQRCode

# ── 全局常量 ─────────────────────────────────────────────
PER_IMAGE_TIMEOUT = 20.0  # 单张图片超时上限(秒)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# 多尺度缩放因子 (按命中率排列: 0.3 > 0.5 > 0.75 > 1.5)
SCALE_FACTORS = [0.3, 0.5, 0.75, 1.5]

# 白边填充像素
PADDING_SIZES = [80, 120]

# ROI 区域定义: 精简版, 优先覆盖右上角高频位置
ROI_REGIONS = [
    (0.60, 0.0, 0.40, 0.35),   # 右上角 (核心命中区)
    (0.55, 0.0, 0.45, 0.40),   # 右上角 (稍大, 覆盖裁切边缘)
    (0.50, 0.0, 0.50, 0.45),   # 右半上部 (宽覆盖)
    (0.0, 0.0, 0.40, 0.35),    # 左上角 (兜底)
]


# ── WeChatQRCode 单例 ───────────────────────────────────
_detector: Optional[WeChatQRCode] = None


def _get_detector() -> WeChatQRCode:
    """延迟初始化 WeChatQRCode 检测器 (单例)。"""
    global _detector
    if _detector is None:
        detect_proto = str(MODELS_DIR / "detect.prototxt")
        detect_model = str(MODELS_DIR / "detect.caffemodel")
        sr_proto = str(MODELS_DIR / "sr.prototxt")
        sr_model = str(MODELS_DIR / "sr.caffemodel")
        for p in [detect_proto, detect_model, sr_proto, sr_model]:
            if not isfile(p):
                raise FileNotFoundError(f"模型文件缺失: {p}")
        _detector = WeChatQRCode(
            detect_proto, detect_model, sr_proto, sr_model
        )
        logger.info("WeChatQRCode 检测器初始化完成")
    return _detector


def _try_detect(img: ndarray) -> Optional[str]:
    """调用 WeChatQRCode 尝试识别, 返回第一个有效结果或 None。"""
    detector = _get_detector()
    results, points = detector.detectAndDecode(img)
    for r in results:
        if r and r.strip():
            return r.strip()
    return None


# ── 预处理函数 ──────────────────────────────────────────


def _add_padding(img: ndarray, pad: int = 80) -> ndarray:
    """为图像四周添加白色边框, 恢复静区 (Quiet Zone)。"""
    return copyMakeBorder(
        img, pad, pad, pad, pad, BORDER_CONSTANT, value=(255, 255, 255)
    )


def _clahe_enhance(img: ndarray) -> ndarray:
    """CLAHE 对比度增强 (仅作用于亮度通道)。"""
    lab = cvtColor(img, COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv_split(lab)
    clahe = createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    enhanced = cv_merge([l_ch, a_ch, b_ch])
    return cvtColor(enhanced, COLOR_LAB2BGR)


def _gamma_correct(img: ndarray, gamma: float = 0.7) -> ndarray:
    """Gamma 校正, gamma < 1 提亮暗部。"""
    inv_gamma = 1.0 / gamma
    table = np_array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return LUT(img, table)


def _sharpen(img: ndarray) -> ndarray:
    """USM 锐化。"""
    blurred = GaussianBlur(img, (0, 0), 3)
    return addWeighted(img, 1.5, blurred, -0.5, 0)


def _adaptive_binarize(img: ndarray) -> ndarray:
    """自适应二值化, 增强低对比度二维码的边缘。"""
    gray = cvtColor(img, COLOR_BGR2GRAY)
    binary = adaptiveThreshold(
        gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY, 51, 10
    )
    return cvtColor(binary, COLOR_GRAY2BGR)


def _resize(img: ndarray, scale: float) -> ndarray:
    """按比例缩放图像。"""
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    interp = INTER_CUBIC if scale > 1.0 else INTER_AREA
    return cv_resize(img, (new_w, new_h), interpolation=interp)


def _extract_roi(
    img: ndarray, roi: tuple[float, float, float, float]
) -> ndarray:
    """从图像中裁剪指定的 ROI 区域。"""
    h, w = img.shape[:2]
    x0 = int(w * roi[0])
    y0 = int(h * roi[1])
    x1 = int(w * (roi[0] + roi[2]))
    y1 = int(h * (roi[1] + roi[3]))
    return img[y0:y1, x0:x1].copy()


def _try_enhanced_detect(crop: ndarray, deadline: float) -> Optional[str]:
    """对裁剪区域尝试多种增强 + 多尺度组合。

    采用分梯队策略，优先尝试高命中率组合。
    """
    # 第一梯队：最强组合 (Gamma + CLAHE)
    enhanced_gclahe = _clahe_enhance(_gamma_correct(crop, 0.7))
    for pad in PADDING_SIZES:
        if perf_counter() > deadline:
            return None
        padded = _add_padding(enhanced_gclahe, pad)
        # 优先试 0.5 缩放 (高频命中)
        for scale in [0.5, 0.75, 1.0]:
            if perf_counter() > deadline:
                return None
            result = _try_detect(_resize(padded, scale))
            if result:
                return result

    # 第二梯队：CLAHE + 锐化 (对边缘清晰但整体偏暗的样本有效)
    enhanced_cs = _sharpen(_clahe_enhance(crop))
    for pad in [100]:
        if perf_counter() > deadline:
            return None
        padded = _add_padding(enhanced_cs, pad)
        for scale in [0.5, 0.75]:
            if perf_counter() > deadline:
                return None
            result = _try_detect(_resize(padded, scale))
            if result:
                return result

    # 第三梯队：极端亮度 (针对极黑单据)
    enhanced_dark = _clahe_enhance(_gamma_correct(crop, 0.3))
    for scale in [0.5, 0.4]:
        if perf_counter() > deadline:
            return None
        result = _try_detect(_resize(_add_padding(enhanced_dark, 120), scale))
        if result:
            return result

    return None


# ── 各层级识别逻辑 ──────────────────────────────────────


def _layer0_raw(img: ndarray, deadline: float) -> Optional[str]:
    """Layer 0: 原图直接识别。"""
    return _try_detect(img)


def _layer1_roi_enhanced(img: ndarray, deadline: float) -> Optional[str]:
    """Layer 1: ROI 裁剪 + 增强 + 多尺度。

    采用 Width-First 宽度优先策略：对所有 ROI 优先尝试最高命中率组合。
    包含 L1 (Standard) 和 L1b (Large Padding 补漏)。
    """
    # 三种 ROI 裁切比例 (x_offset, y_offset, width_ratio, height_ratio)
    # 均针对右上角区域, 从小到大覆盖不同的裁切程度
    roi_configs = [
        (0.60, 0.0, 0.40, 0.35),
        (0.55, 0.0, 0.45, 0.40),
        (0.50, 0.0, 0.50, 0.45),
    ]
    pad_sizes = [80, 120]

    # 增强函数 + 对应缩放因子组合, 按实测命中率从高到低排列
    # 宽度优先: 每种增强先对所有 ROI 尝试一遍, 再换下一种增强
    enhancements = [
        # Gamma(0.7) + CLAHE: 对绝大多数灰底扫描件命中率最高
        (lambda x: _clahe_enhance(_gamma_correct(x, 0.7)), [0.5, 0.75, 1.0]),
        # CLAHE + 锐化: 对模糊但对比度尚可的样本有效
        (lambda x: _sharpen(_clahe_enhance(x)), [0.5, 0.75]),
        # 纯 CLAHE: 轻量兜底
        (lambda x: _clahe_enhance(x), [0.5, 0.75, 1.0]),
        # Gamma(0.3) + CLAHE: 极端压暗, 针对灰底灰码的低对比度场景
        (lambda x: _clahe_enhance(_gamma_correct(x, 0.3)), [0.3, 0.4, 0.5]),
        # Gamma(0.5) + CLAHE: 介于 0.3 和 0.7 之间的补充档位
        (lambda x: _clahe_enhance(_gamma_correct(x, 0.5)), [0.5, 0.4]),
    ]

    # 预裁剪 ROI
    rois = [(_extract_roi(img, r), r) for r in roi_configs]

    # 宽度优先遍历
    for e_fn, scales in enhancements:
        for crop, r_coords in rois:
            if crop.size == 0:
                continue
            enhanced = e_fn(crop)
            for pad in pad_sizes:
                if perf_counter() > deadline:
                    return None
                padded = _add_padding(enhanced, pad)
                for scale in scales:
                    if perf_counter() > deadline:
                        return None
                    result = _try_detect(_resize(padded, scale))
                    if result:
                        logger.debug(
                            f"  Layer1 命中: ROI={r_coords}, "
                            f"scale={scale}, pad={pad}"
                        )
                        return result

    # Layer 1b — 针对定位角被边缘严重裁切的样本:
    # 使用 200px 超大白边 + 极端压暗 (Gamma 0.3) 恢复被截断的静区和角点
    l1b_rois = [(0.70, 0.0, 0.30, 0.20), (0.45, 0.0, 0.55, 0.40)]
    for roi_coords in l1b_rois:
        if perf_counter() > deadline:
            return None
        crop = _extract_roi(img, roi_coords)
        if crop.size == 0:
            continue
        enhanced = _clahe_enhance(_gamma_correct(crop, 0.3))
        padded = _add_padding(enhanced, 200)
        for scale in [0.3, 0.4, 0.5]:
            if perf_counter() > deadline:
                return None
            result = _try_detect(_resize(padded, scale))
            if result:
                logger.debug(
                    f"  Layer1b 命中: ROI={roi_coords}, "
                    f"scale={scale}, pad=200"
                )
                return result

    return None


def _layer2_fullimg_enhance(img: ndarray, deadline: float) -> Optional[str]:
    """Layer 2: 全图增强 (CLAHE + 锐化) + padding + 多尺度。"""
    enhanced = _sharpen(_clahe_enhance(img))
    for pad in PADDING_SIZES:
        if perf_counter() > deadline:
            return None
        padded = _add_padding(enhanced, pad)
        result = _try_detect(padded)
        if result:
            return result
        for scale in SCALE_FACTORS:
            if perf_counter() > deadline:
                return None
            scaled = _resize(padded, scale)
            result = _try_detect(scaled)
            if result:
                logger.debug(f"  Layer2 命中: pad={pad}, scale={scale}")
                return result
    return None


def _layer3_ctf_repair(img: ndarray, deadline: float) -> Optional[str]:
    """Layer 3: CTF 风格 Finder Pattern 修复 (由 qr_repair 模块提供)。

    将 pipeline 已有的 detector 单例传入 repair_and_detect(), 避免重复加载模型。
    如果 qr_repair 模块不可用则静默跳过本层。
    """
    try:
        from src.qr_repair import repair_and_detect
    except ImportError as e:
        logger.debug(f"  Layer3 跳过: qr_repair 模块导入失败 {e}")
        return None
    # 传入已初始化的 detector 单例, 节省 ~1s 模型加载时间
    return repair_and_detect(img, deadline=deadline, detector=_get_detector())


# ── 流水线主入口 ─────────────────────────────────────────

# 按成本递增排列的层级列表
_LAYERS = [
    ("Layer0_Raw", _layer0_raw),
    ("Layer1_ROI+Enhanced", _layer1_roi_enhanced),
    ("Layer2_FullImg+CLAHE", _layer2_fullimg_enhance),
    ("Layer3_CTF_Repair", _layer3_ctf_repair),
]


def _run_pipeline_sync(img: ndarray) -> tuple[Optional[str], str, float]:
    """同步执行流水线(在工作线程中运行)。

    Returns:
        (识别结果, 命中层级, 总耗时秒)
    """
    t_start = perf_counter()
    deadline = t_start + PER_IMAGE_TIMEOUT - 0.5  # 留 0.5s 缓冲
    for layer_name, layer_fn in _LAYERS:
        if perf_counter() > deadline:
            logger.warning(f"  时间不足, 跳过 {layer_name}")
            break
        t_layer = perf_counter()
        try:
            result = layer_fn(img, deadline)
        except Exception as e:
            logger.error(f"  {layer_name} 异常: {e}")
            continue
        dt = perf_counter() - t_layer
        if result:
            total = perf_counter() - t_start
            logger.info(
                f"  {layer_name} 识别成功 ({dt:.3f}s, 总计 {total:.3f}s)"
            )
            return result, layer_name, total
        else:
            logger.debug(f"  {layer_name} 未识别 ({dt:.3f}s)")
    total = perf_counter() - t_start
    return None, "FAILED", total


def scan_image(image_path: str | Path) -> dict:
    """扫描单张图片, 返回识别结果字典。

    包含 20s 硬超时熔断保护。

    Returns:
        {
            "file": 文件名,
            "result": 识别内容 | None,
            "layer": 命中层级,
            "time_s": 耗时(秒),
            "status": "success" | "timeout" | "failed" | "error"
        }
    """
    image_path = Path(image_path)
    logger.info(f"处理: {image_path.name}")

    img = imread(str(image_path))
    if img is None:
        logger.error(f"  无法读取图像: {image_path}")
        return {
            "file": image_path.name,
            "result": None,
            "layer": "N/A",
            "time_s": 0.0,
            "status": "error",
        }

    t0 = perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_pipeline_sync, img)
        try:
            result, layer, elapsed = future.result(timeout=PER_IMAGE_TIMEOUT)
            status = "success" if result else "failed"
        except FuturesTimeoutError:
            elapsed = perf_counter() - t0
            logger.warning(
                f"  超时熔断! ({elapsed:.1f}s > {PER_IMAGE_TIMEOUT}s)"
            )
            future.cancel()
            return {
                "file": image_path.name,
                "result": None,
                "layer": "TIMEOUT",
                "time_s": round(elapsed, 3),
                "status": "timeout",
            }
        except Exception as e:
            elapsed = perf_counter() - t0
            logger.error(f"  流水线异常: {e}")
            return {
                "file": image_path.name,
                "result": None,
                "layer": "ERROR",
                "time_s": round(elapsed, 3),
                "status": "error",
            }

    return {
        "file": image_path.name,
        "result": result,
        "layer": layer,
        "time_s": round(elapsed, 3),
        "status": status,
    }


def scan_directory(dir_path: str | Path) -> list[dict]:
    """批量扫描目录下所有图片, 返回结果列表。"""
    dir_path = Path(dir_path)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    files = sorted(
        f for f in dir_path.iterdir() if f.suffix.lower() in image_exts
    )
    logger.info(f"共发现 {len(files)} 张待扫描图片")

    results = []
    for f in files:
        r = scan_image(f)
        results.append(r)

    # 汇总统计
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    timeout = sum(1 for r in results if r["status"] == "timeout")
    errors = sum(1 for r in results if r["status"] == "error")
    avg_time = sum(r["time_s"] for r in results) / total if total else 0

    logger.info("=" * 60)
    logger.info(f"扫描完成: {total} 张图片")
    logger.info(f"  成功: {success}/{total} ({100*success/total:.1f}%)")
    logger.info(f"  失败: {failed}, 超时: {timeout}, 错误: {errors}")
    logger.info(f"  平均耗时: {avg_time:.3f}s/张")
    logger.info("=" * 60)

    return results

"""
QR 码增强识别 — 独立调试入口 (qr_repair.py)

本模块同时承担两个职责:
  1. 作为独立可执行脚本: 对 failed/ 目录下的样本批量跑分层识别策略,
     用于算法迭代时的离线效果评估 (`uv run python -m src.qr_repair`)。
  2. 提供 repair_and_detect() 函数, 供 src/pipeline.py 的 Layer 3
     (CTF 修复层) 调用。

分层递进 + 早停 (Short-Circuit) 架构:
  Layer 0: 原图直接识别 (最快, ~50ms)
  Layer 1: ROI 裁剪 + 多种增强 + 多尺度 (核心命中区, 命中率 95%+)
  Layer 1b: 大 padding 补漏 (针对边缘严重裁切样本)
  Layer 2: Finder Pattern 几何修复 + 增强 + 多尺度
  Layer 3: 全图多尺度兜底
"""

from os import listdir
from os.path import join as path_join
from pathlib import Path
from sys import argv as sys_argv, exit as sys_exit, stderr as sys_stderr
from time import perf_counter
from typing import Optional

from cv2 import imread
from numpy import ndarray
from loguru import logger

from src.preprocess import (
    add_white_padding,
    apply_clahe,
    binarize_otsu,
    crop_roi,
    denoise_then_binarize,
    gamma_correct,
    rescale,
    sharpen,
)
from src.finder_repair import repair_finder_patterns

# WeChatQRCode 模型类 — contrib 子模块, 显式导入
from cv2.wechat_qrcode import WeChatQRCode


# ── 配置常量 ─────────────────────────────────────────────

# 使用绝对路径, 保证无论从哪个工作目录调用都能找到模型文件
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = str(_PROJECT_ROOT / "models")
FAILED_DIR = str(_PROJECT_ROOT / "failed")

DETECT_PROTOTXT = path_join(MODELS_PATH, "detect.prototxt")
DETECT_CAFFEMODEL = path_join(MODELS_PATH, "detect.caffemodel")
SR_PROTOTXT = path_join(MODELS_PATH, "sr.prototxt")
SR_CAFFEMODEL = path_join(MODELS_PATH, "sr.caffemodel")

# 单张图片处理时间硬上限 (秒)
PER_IMAGE_TIMEOUT = 20.0

# 缩放因子按实测命中率排序 (高频在前, 早停早收益)
PRIORITY_SCALES = [0.5, 1.0, 0.75, 1.5, 2.0]
# 精简版 — 用于低优先级组合, 节省时间预算给后续层级
FAST_SCALES = [0.5, 0.75, 1.0]


# ── 检测器初始化 ─────────────────────────────────────────


# 模块级单例 — 首次调用 init_detector() 时创建, 后续复用
_detector: Optional[WeChatQRCode] = None


def init_detector() -> WeChatQRCode:
    """延迟初始化并缓存 WeChatQRCode 检测器 (单例)。

    首次调用时加载 detect + sr 两组 Caffe 模型文件 (~1s),
    后续调用直接返回已有实例, 避免重复的磁盘 I/O 和模型解析开销。
    """
    global _detector
    if _detector is None:
        _detector = WeChatQRCode(
            DETECT_PROTOTXT, DETECT_CAFFEMODEL, SR_PROTOTXT, SR_CAFFEMODEL
        )
        logger.info("WeChatQRCode detector initialized.")
    return _detector


# ── 识别原语 ─────────────────────────────────────────────


def try_decode(detector: WeChatQRCode, img: ndarray) -> Optional[str]:
    """对单张图调用检测器, 返回第一个非空识别结果或 None。"""
    res, points = detector.detectAndDecode(img)
    if res and len(res) > 0 and res[0]:
        return res[0]
    return None


def try_with_scales(detector, img, label, scales=None, deadline=None):
    """对一张图按多个缩放因子顺序尝试识别, 命中即返回。

    Args:
        detector: WeChatQRCode 实例
        img:      待识别图像
        label:    日志/返回值中标注的策略名 (用于事后分析命中分布)
        scales:   缩放因子列表 (None 则用 PRIORITY_SCALES)
        deadline: 截止时间戳 (perf_counter), 超时立即返回 (None, None)

    Returns:
        (result, label_with_scale) 或 (None, None)
    """
    if scales is None:
        scales = PRIORITY_SCALES
    for scale in scales:
        if deadline and perf_counter() >= deadline:
            return None, None
        scaled = rescale(img, scale)
        if scaled is None:
            continue
        result = try_decode(detector, scaled)
        if result:
            return result, f"{label}_x{scale}"
    return None, None


def _apply_enhancement(roi: ndarray, pad: int, enhance_name: str) -> ndarray:
    """对 ROI 添加白边后按策略名应用增强算子。

    使用字符串名称调度而非函数引用, 目的是让 l1_combos 列表保持可读性,
    同时使日志中打印的策略名称与代码中的枚举保持一致。

    支持的策略名 (enhance_name):
        "gamma_clahe"   — Gamma(0.6) + CLAHE: 通用场景首选, 适合大多数灰底扫描件
        "gamma03_clahe" — Gamma(0.3) + CLAHE: 极端压暗, 针对灰底灰码低对比度场景
        "gamma04_clahe" — Gamma(0.4) + CLAHE: 中强度压暗
        "gamma05_clahe" — Gamma(0.5) + CLAHE: 中等压暗
        "gamma08_clahe" — Gamma(0.8) + CLAHE: 轻微压暗, 针对印刷偏深样本
        "clahe+sharp"   — CLAHE + USM 锐化: 模糊边缘增强
        "clahe"         — 纯 CLAHE: 轻量对比度增强
        "padded"        — 仅加白边, 不做增强: 用于原图即可识别但需静区的样本
        "otsu"          — Otsu 全局二值化: 背景均匀时有效
        "denoise_bin"   — 降噪 + 二值化: 噪点干扰严重场景
    """
    padded = add_white_padding(roi, pad)
    if enhance_name == "gamma_clahe":
        return apply_clahe(gamma_correct(padded, 0.6))
    elif enhance_name == "gamma03_clahe":
        return apply_clahe(gamma_correct(padded, 0.3))
    elif enhance_name == "gamma04_clahe":
        return apply_clahe(gamma_correct(padded, 0.4))
    elif enhance_name == "gamma05_clahe":
        return apply_clahe(gamma_correct(padded, 0.5))
    elif enhance_name == "gamma08_clahe":
        return apply_clahe(gamma_correct(padded, 0.8))
    elif enhance_name == "clahe+sharp":
        return sharpen(apply_clahe(padded))
    elif enhance_name == "clahe":
        return apply_clahe(padded)
    elif enhance_name == "padded":
        return padded
    elif enhance_name == "otsu":
        return binarize_otsu(padded)
    elif enhance_name == "denoise_bin":
        return denoise_then_binarize(padded)
    return padded


# ── 主流水线 ─────────────────────────────────────────────


def process_image(detector, img_path):
    """分层递进 + 早停策略处理单张图片。

    Returns:
        (识别结果|None, 耗时秒, 策略名/失败原因)
    """
    img = imread(img_path)
    if img is None:
        logger.warning(f"Cannot read: {img_path}")
        return None, 0, "read_error"

    t0 = perf_counter()
    deadline = t0 + PER_IMAGE_TIMEOUT

    # ===== Layer 0: 原图快速尝试 =====
    result = try_decode(detector, img)
    if result:
        return result, perf_counter() - t0, "L0_original"

    padded_full = add_white_padding(img, 100)
    result = try_decode(detector, padded_full)
    if result:
        return result, perf_counter() - t0, "L0_original+pad"

    # ===== Layer 1: ROI 裁剪 + 增强 + 多尺度 =====
    # 宽度优先策略: 先对所有 ROI 试最高命中率组合, 再扩展到低概率组合,
    # 命中率最高的 (gamma_clahe x0.5) 会最早被尝试, 命中即终止节省时间。
    roi_configs = [(0.25, 0.15), (0.30, 0.18), (0.35, 0.20), (0.40, 0.25)]
    pad_sizes = [80, 120]

    l1_combos = [
        # === 第一梯队: 高命中率 ===
        ("gamma_clahe", [0.5]),
        ("gamma_clahe", [0.75, 1.0]),
        ("clahe+sharp", [0.5, 0.75]),
        ("clahe", [0.5, 0.75, 1.0]),
        ("gamma_clahe", [1.5, 2.0]),
        # === 第二梯队: 困难图 (小缩放 + 强 gamma) ===
        ("gamma_clahe", [0.4, 0.3]),
        ("gamma03_clahe", [0.5, 0.4, 0.3, 0.8, 0.9]),
        ("gamma04_clahe", [0.5, 0.4, 0.3]),
        ("gamma05_clahe", [0.5, 0.4]),
        ("gamma08_clahe", [0.5, 0.4]),
        # === 第三梯队: 兜底补充 ===
        ("clahe+sharp", [1.0, 0.4, 0.9]),
        ("padded", [0.5, 0.4]),
    ]

    # 预裁剪 ROI 复用 (避免对每个增强组合都重复裁剪)
    rois = [(w_r, h_r, crop_roi(img, w_r, h_r)) for w_r, h_r in roi_configs]

    for enh, scales in l1_combos:
        for w_r, h_r, roi in rois:
            for pad in pad_sizes:
                if perf_counter() >= deadline:
                    return None, perf_counter() - t0, "timeout_L1"
                enhanced = _apply_enhancement(roi, pad, enh)
                res, tag = try_with_scales(
                    detector, enhanced,
                    f"L1_roi({w_r},{h_r})_p{pad}_{enh}",
                    scales=scales,
                    deadline=deadline,
                )
                if res:
                    return res, perf_counter() - t0, tag

    # ===== Layer 1b: 大 padding 补漏 (针对严重裁切图) =====
    # 单独一层的原因: pad=200 比 pad=80/120 慢得多, 分离后可在 L1
    # 高频组合都失败时才付出此成本。
    l1b_rois = [(0.30, 0.18), (0.45, 0.30)]
    l1b_pads = [200]
    l1b_combos = [
        ("gamma03_clahe", [0.3, 0.4, 0.5]),
        ("gamma04_clahe", [0.3, 0.4, 0.5]),
        ("gamma_clahe", [0.3, 0.4, 0.5]),
    ]
    l1b_roi_imgs = [
        (w_r, h_r, crop_roi(img, w_r, h_r)) for w_r, h_r in l1b_rois
    ]
    for enh, scales in l1b_combos:
        for w_r, h_r, roi in l1b_roi_imgs:
            for pad in l1b_pads:
                if perf_counter() >= deadline:
                    return None, perf_counter() - t0, "timeout_L1b"
                enhanced = _apply_enhancement(roi, pad, enh)
                res, tag = try_with_scales(
                    detector, enhanced,
                    f"L1b_roi({w_r},{h_r})_p{pad}_{enh}",
                    scales=scales,
                    deadline=deadline,
                )
                if res:
                    return res, perf_counter() - t0, tag

    # ===== Layer 2: Finder Pattern 修复 + 增强 + 多尺度 =====
    # 当增强手段都无效时, 大概率是定位角受损 (印章/裁切),
    # 此层用几何补全后再尝试识别。
    repair_rois = [(0.25, 0.15), (0.35, 0.20), (0.40, 0.25)]
    repair_pads = [80, 120]
    repair_enhances = [
        "gamma_clahe", "clahe+sharp", "clahe", "padded", "otsu", "denoise_bin",
    ]

    for w_r, h_r in repair_rois:
        if perf_counter() >= deadline:
            return None, perf_counter() - t0, "timeout_L2"
        roi = crop_roi(img, w_r, h_r)

        for pad in repair_pads:
            if perf_counter() >= deadline:
                return None, perf_counter() - t0, "timeout_L2"

            padded = add_white_padding(roi, pad)

            # 对原图 / gclahe / clahe 三个版本分别尝试 Finder 修复,
            # 因为 Finder 检测器对前置增强敏感
            repair_inputs = [
                ("raw", padded),
                ("gclahe", apply_clahe(gamma_correct(padded))),
                ("clahe", apply_clahe(padded)),
            ]

            for ri_label, ri_img in repair_inputs:
                if perf_counter() >= deadline:
                    return None, perf_counter() - t0, "timeout_L2"

                repaired, found, fixed = repair_finder_patterns(ri_img)

                if fixed > 0:
                    logger.debug(
                        f"Finder repair on {ri_label}: found={found}, "
                        f"fixed={fixed} for roi({w_r},{h_r}) pad={pad}"
                    )

                    # 修复成功后再尝试各种增强 + 多尺度组合
                    for enh in repair_enhances:
                        if perf_counter() >= deadline:
                            return None, perf_counter() - t0, "timeout_L2"

                        if enh == "padded":
                            enhanced = repaired
                        elif enh == "gamma_clahe":
                            enhanced = apply_clahe(gamma_correct(repaired))
                        elif enh == "clahe+sharp":
                            enhanced = sharpen(apply_clahe(repaired))
                        elif enh == "clahe":
                            enhanced = apply_clahe(repaired)
                        elif enh == "otsu":
                            enhanced = binarize_otsu(repaired)
                        elif enh == "denoise_bin":
                            enhanced = denoise_then_binarize(repaired)
                        else:
                            enhanced = repaired

                        res, tag = try_with_scales(
                            detector, enhanced,
                            f"L2_repair({ri_label})_roi({w_r},{h_r})_p{pad}_{enh}",
                            deadline=deadline,
                        )
                        if res:
                            return res, perf_counter() - t0, tag

            # 二值化兜底 (无修复时也尝试一下)
            for enh in ["otsu", "denoise_bin"]:
                if perf_counter() >= deadline:
                    return None, perf_counter() - t0, "timeout_L2"
                enhanced = _apply_enhancement(roi, pad, enh)
                res, tag = try_with_scales(
                    detector, enhanced,
                    f"L2_roi({w_r},{h_r})_p{pad}_{enh}",
                    deadline=deadline,
                )
                if res:
                    return res, perf_counter() - t0, tag

    # ===== Layer 3: 全图多尺度兜底 =====
    # 放弃 ROI 裁切, 对整图进行多倍缩放 + CLAHE 增强 (gamma 默认 0.6)
    for scale in [1.5, 2.0, 0.5, 3.0]:
        if perf_counter() >= deadline:
            return None, perf_counter() - t0, "timeout_L3"
        resized = rescale(img, scale)
        if resized is None:
            continue
        resized_pad = add_white_padding(resized, 120)
        result = try_decode(detector, resized_pad)
        if result:
            return result, perf_counter() - t0, f"L3_full_x{scale}"

        if perf_counter() >= deadline:
            return None, perf_counter() - t0, "timeout_L3"
        enhanced = apply_clahe(gamma_correct(resized_pad))  # gamma=0.6 (preprocess.py 默认值)
        result = try_decode(detector, enhanced)
        if result:
            return result, perf_counter() - t0, f"L3_full_gclahe_x{scale}"

    elapsed = perf_counter() - t0
    return None, elapsed, "all_failed"


# ── pipeline.py 的 Layer 3 桥接入口 ──────────────────────


def repair_and_detect(img: ndarray, deadline: float = None,
                      detector: Optional[WeChatQRCode] = None) -> Optional[str]:
    """供 src/pipeline.py 的 Layer 3 调用的几何修复 + 识别入口。

    当 Layer 0~2 均识别失败时调用。聚焦于"定位角受损"这一特定失败模式:
    对 ROI 子图尝试 Finder Pattern 检测 → 缺失角推算 → 补绘 → 再识别。

    工作流程 (对每个 ROI + padding 组合):
      1. crop_roi()               — 裁取右上角子图 (单据二维码的高频位置)
      2. add_white_padding()      — 补充静区, 恢复被裁切的 Quiet Zone
      3. repair_finder_patterns() — 检测现有定位角, 推算并绘制缺失的第三角
      4. 修复成功 (fixed > 0):   对修复图尝试直接识别、多尺度和 CLAHE 增强

    Args:
        img:      待识别的 BGR 图像 (numpy ndarray)
        deadline: perf_counter() 时间戳截止值, 超过时立即返回 None
        detector: 可选的外部 WeChatQRCode 实例 (由 pipeline.py 传入以复用单例,
                  避免重复加载模型); 不传时使用本模块自身的单例

    Returns:
        识别到的二维码字符串, 或 None (修复后仍无法识别 / 超时)
    """
    if detector is None:
        detector = init_detector()

    # 右上角三种裁切比例 (w_ratio x h_ratio), 从紧凑到宽松依次尝试
    repair_rois = [(0.25, 0.15), (0.35, 0.20), (0.40, 0.25)]
    repair_pads = [100, 150]

    for w_r, h_r in repair_rois:
        if deadline and perf_counter() >= deadline:
            return None

        roi = crop_roi(img, w_r, h_r)
        if roi.size == 0:
            continue

        for pad in repair_pads:
            if deadline and perf_counter() >= deadline:
                return None

            padded = add_white_padding(roi, pad)

            # 对原图和 CLAHE 增强版分别做修复 — 低对比度样本在增强后检出率更高
            repair_inputs = [
                ("raw", padded),
                ("gclahe", apply_clahe(gamma_correct(padded, 0.6))),
            ]

            for _, ri_img in repair_inputs:
                if deadline and perf_counter() >= deadline:
                    return None

                repaired, _, fixed = repair_finder_patterns(ri_img)

                if fixed > 0:
                    # 定位角修复成功, 按由简到繁尝试识别
                    res = try_decode(detector, repaired)
                    if res:
                        return res

                    for scale in [0.5, 0.75, 1.0]:
                        if deadline and perf_counter() >= deadline:
                            return None
                        scaled = rescale(repaired, scale)
                        res = try_decode(detector, scaled)
                        if res:
                            return res

                    enhanced = apply_clahe(repaired)
                    res = try_decode(detector, enhanced)
                    if res:
                        return res

    return None


# ── 独立调试入口 ─────────────────────────────────────────


def main():
    """脚本入口: 批量评估 failed/ 目录下所有样本的识别效果。"""
    logger.remove()
    logger.add(sys_stderr, level="DEBUG")
    logger.add("qr_repair.log", rotation="5 MB", level="DEBUG")

    detector = init_detector()

    image_files = sorted([
        f for f in listdir(FAILED_DIR)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])
    logger.info(f"Found {len(image_files)} images in {FAILED_DIR}/")

    success_count = 0
    total_time = 0.0
    failed_list = []

    for img_name in image_files:
        img_path = path_join(FAILED_DIR, img_name)
        result, elapsed, strategy = process_image(detector, img_path)
        total_time += elapsed

        if result:
            success_count += 1
            logger.success(
                f"[{img_name}] OK | strategy={strategy} | "
                f"result={result[:60]} | {elapsed:.3f}s"
            )
        else:
            failed_list.append(img_name)
            if "timeout" in strategy:
                logger.error(f"[{img_name}] TIMEOUT ({strategy}) | {elapsed:.3f}s")
            else:
                logger.error(f"[{img_name}] FAILED | {elapsed:.3f}s")

    # 汇总报告
    total = len(image_files)
    rate = success_count / total * 100 if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0

    logger.info("=" * 60)
    logger.info(
        f"Total: {total} | Success: {success_count} | "
        f"Rate: {rate:.1f}% | Avg: {avg_time:.3f}s | "
        f"Total: {total_time:.1f}s"
    )
    if failed_list:
        logger.warning(f"Failed images: {failed_list}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

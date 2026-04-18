"""
QR 码 Finder Pattern (定位角) 检测与修复模块。

QR 码标准定义了 3 个位于左上、右上、左下角的 Finder Pattern (定位角),
其作用是让扫描器快速定位二维码的位置和方向。一旦其中任何一个定位角
受损 (印刷模糊、印章遮挡、边缘裁切), WeChatQRCode 检测器就会失败。

本模块实现了 CTF (Capture The Flag) 比赛中常见的"定位角补全"思路:

  ┌────────────────────────────────────────────────────────┐
  │ 1. detect_finder_patterns()                            │
  │    用两种独立方法找出图中已存在的定位角候选:           │
  │      a) 嵌套轮廓法 — 找三层嵌套的正方形                │
  │      b) 逐行扫描法 — 找符合 1:1:3:1:1 比例的黑白条纹    │
  │    最后做空间聚类去重。                                │
  │                                                        │
  │ 2. infer_missing_corner()                              │
  │    根据已知的两个定位角, 利用 QR 码的几何关系          │
  │    (TL-TR 水平, TL-BL 垂直) 反推出缺失角的坐标。       │
  │                                                        │
  │ 3. draw_finder_pattern()                               │
  │    在缺失位置绘制完美的 7x7 标准定位角模板, 让检测器   │
  │    认为该位置存在合法的 Finder Pattern。               │
  └────────────────────────────────────────────────────────┘

入口函数: repair_finder_patterns()
"""

from numpy import (
    any as np_any,
    array as np_array,
    percentile as np_percentile,
)
from cv2 import (
    ADAPTIVE_THRESH_GAUSSIAN_C,
    CHAIN_APPROX_SIMPLE,
    COLOR_BGR2GRAY,
    RETR_TREE,
    THRESH_BINARY,
    THRESH_OTSU,
    adaptiveThreshold,
    approxPolyDP,
    arcLength,
    boundingRect,
    contourArea,
    createCLAHE,
    cvtColor,
    findContours,
    rectangle,
    threshold,
)
from loguru import logger


# ── 内部辅助函数 ─────────────────────────────────────────


def _check_ratio(segments, tolerance: float = 0.5) -> bool:
    """验证 5 段 run-length 是否符合 Finder Pattern 的 1:1:3:1:1 模块比例。

    QR 码穿过 Finder Pattern 中心的扫描线必然呈现
    "黑-白-黑-白-黑" 的 5 段, 长度比例为 1:1:3:1:1 (共 7 个模块)。

    Args:
        segments:  5 个连续段的像素长度
        tolerance: 容差比例 (0.5 = 允许 ±50% 偏差, 应对模块尺寸误差)

    Returns:
        True 表示比例符合 Finder Pattern 特征。
    """
    total = sum(segments)
    if total == 0:
        return False
    module = total / 7.0
    expected = [module, module, 3 * module, module, module]
    for seg, exp in zip(segments, expected):
        if abs(seg - exp) > tolerance * exp:
            return False
    return True


def _find_nested_squares(contours, hierarchy):
    """从轮廓集合中找出三层嵌套的正方形 — 标准 Finder Pattern 的拓扑特征。

    Finder Pattern 在轮廓层级上呈现 外正方形 → 白带正方形 → 内核正方形
    的三层嵌套结构。本函数遍历所有轮廓, 识别外层方形且至少有一个
    嵌套子方形的候选, 并返回中心坐标和边长。

    面积比启发式: 内框/外框 ∈ (0.15, 0.65), 涵盖 (3/7)^2≈0.18 到 (5/7)^2≈0.51。

    Returns:
        list of (cx, cy, side) — 候选定位角的中心坐标和边长。
    """
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    candidates = []

    for i, cnt in enumerate(contours):
        # 过滤面积过小的轮廓 (噪声)
        area = contourArea(cnt)
        if area < 100:
            continue

        peri = arcLength(cnt, True)
        approx = approxPolyDP(cnt, 0.05 * peri, True)
        if len(approx) != 4:
            continue

        # 长宽比近似 1:1 (允许 ±30% 形变, 应对扫描畸变)
        x, y, w, h = boundingRect(approx)
        aspect = w / float(h) if h > 0 else 0
        if not (0.7 < aspect < 1.3):
            continue

        # 必须有嵌套的子轮廓
        child_idx = hierarchy[i][2]
        if child_idx == -1:
            continue

        child_cnt = contours[child_idx]
        child_area = contourArea(child_cnt)
        if child_area < 20:
            continue

        child_peri = arcLength(child_cnt, True)
        child_approx = approxPolyDP(child_cnt, 0.05 * child_peri, True)
        if len(child_approx) != 4:
            continue

        # 内外面积比启发式: 0.15 < ratio < 0.65 覆盖嵌套 Finder Pattern
        ratio = child_area / area if area > 0 else 0
        if 0.15 < ratio < 0.65:
            cx = x + w // 2
            cy = y + h // 2
            candidates.append((cx, cy, max(w, h)))

    return candidates


def _scan_line_detect(gray):
    """逐行扫描寻找 1:1:3:1:1 模式 — Finder Pattern 的几何不变量。

    嵌套轮廓法在轮廓不闭合时会失败, 此扫描线法作为补充策略。
    每隔若干行 (h//60) 取一行二值化数据, run-length 编码后,
    在所有 BWBWB 段中寻找符合比例的窗口。

    Returns:
        聚类去重后的候选中心点列表 (cx, cy, total_length)。
    """
    h, w = gray.shape
    _, bw = threshold(gray, 0, 255, THRESH_BINARY + THRESH_OTSU)

    centers = []
    step = max(1, h // 60)

    for row in range(0, h, step):
        line = bw[row]
        runs = []
        current_val = line[0]
        run_len = 1

        for col in range(1, w):
            if line[col] == current_val:
                run_len += 1
            else:
                runs.append((current_val, run_len, col - run_len))
                current_val = line[col]
                run_len = 1
        runs.append((current_val, run_len, w - run_len))

        # 滑动窗口: 寻找 B-W-B-W-B (0-255-0-255-0) 5 连段
        for i in range(len(runs) - 4):
            seg = runs[i:i + 5]
            vals = [s[0] for s in seg]
            if vals == [0, 255, 0, 255, 0]:
                lengths = [s[1] for s in seg]
                if _check_ratio(lengths):
                    total_len = sum(lengths)
                    start_x = seg[0][2]
                    cx = start_x + total_len // 2
                    centers.append((cx, row, total_len))

    return _cluster_points(centers)


def _cluster_points(points, min_dist: int = 20):
    """简易空间聚类: 距离小于阈值的候选点合并为一个中心。

    嵌套轮廓法 + 扫描线法会对同一个 Finder Pattern 产生多个候选,
    通过 O(N^2) 的贪心聚类去重, 避免后续推算缺失角时混淆位置。

    Args:
        points:   list of (cx, cy, size)
        min_dist: 聚类阈值 (像素)

    Returns:
        合并后的聚类中心列表。
    """
    if not points:
        return []

    points = sorted(points, key=lambda p: (p[0], p[1]))
    clusters = []
    used = [False] * len(points)

    for i in range(len(points)):
        if used[i]:
            continue
        cx, cy, cs = points[i][0], points[i][1], points[i][2]
        count = 1
        used[i] = True

        for j in range(i + 1, len(points)):
            if used[j]:
                continue
            dx = abs(points[j][0] - cx / count)
            dy = abs(points[j][1] - cy / count)
            if dx < min_dist and dy < min_dist:
                cx += points[j][0]
                cy += points[j][1]
                cs += points[j][2]
                count += 1
                used[j] = True

        clusters.append((cx // count, cy // count, cs // count))

    return clusters


# ── 公开 API ─────────────────────────────────────────────


def detect_finder_patterns(img):
    """检测图像中的 QR 码 Finder Pattern。

    流程:
      1. 转灰度
      2. CLAHE 增强 (灰底单据上原始定位角对比度往往不够)
      3. 对 [增强图, 原图] 分别用 [Otsu, 自适应] 二值化
      4. 对每种二值化结果运行 嵌套轮廓法 + 扫描线法
      5. 取所有候选并集后聚类, 最多保留 3 个最大的候选

    Returns:
        list of (cx, cy, size) — 最多 3 个 Finder Pattern 的中心和尺寸。
    """
    gray = cvtColor(img, COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # CLAHE 增强后再检测, 提升灰底/低对比度场景的检出率
    clahe = createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    all_candidates = []
    for src in [enhanced, gray]:
        # Otsu 全局二值化
        _, bw = threshold(src, 0, 255, THRESH_BINARY + THRESH_OTSU)
        contours, hierarchy = findContours(
            bw, RETR_TREE, CHAIN_APPROX_SIMPLE
        )
        all_candidates.extend(_find_nested_squares(contours, hierarchy))
        all_candidates.extend(_scan_line_detect(src))

        # 自适应高斯阈值 (针对不均匀光照)
        bw_adapt = adaptiveThreshold(
            src, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
            THRESH_BINARY, 51, 15
        )
        contours2, hierarchy2 = findContours(
            bw_adapt, RETR_TREE, CHAIN_APPROX_SIMPLE
        )
        all_candidates.extend(_find_nested_squares(contours2, hierarchy2))

    merged = _cluster_points(all_candidates, min_dist=30)

    # 限制最多 3 个 (QR 标准只有 3 个 Finder Pattern)
    if len(merged) > 3:
        merged.sort(key=lambda p: p[2], reverse=True)
        merged = merged[:3]

    return merged


def infer_missing_corner(found_corners):
    """根据已知的 2 个定位角推算第 3 个缺失角的位置。

    QR 码三个 Finder Pattern 的几何关系 (假设无旋转):
      - TL (左上) 是直角顶点
      - TR (右上) = TL + (dx, 0)  水平相邻
      - BL (左下) = TL + (0, dy)  垂直相邻
      - 由于 QR 码是正方形, dx ≈ dy

    判定逻辑:
      - 两点水平排列  (dx >> dy): 已知 TL+TR, 缺失 BL
      - 两点垂直排列  (dy >> dx): 已知 TL+BL, 缺失 TR
      - 两点对角排列  (dx ≈ dy):  已知 TL+BR 或 TR+BL, 推算 TL/TR

    Args:
        found_corners: 已检测到的 Finder Pattern 列表 (cx, cy, size)

    Returns:
        (corner_type, (cx, cy, size)) 其中 corner_type ∈ {'TL', 'TR', 'BL'}
        若信息不足或已检测到 3 个角则返回 None。
    """
    if len(found_corners) < 2:
        return None
    if len(found_corners) >= 3:
        # 三个角都齐了, 不需要推算
        return None

    p1, p2 = found_corners[0], found_corners[1]
    avg_size = (p1[2] + p2[2]) // 2

    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])

    if dx > dy * 1.5:
        # 水平相邻 → 已知 TL+TR, 推算 BL
        tl = p1 if p1[0] < p2[0] else p2
        tr = p2 if p1[0] < p2[0] else p1
        bl_x = tl[0]
        bl_y = tl[1] + (tr[0] - tl[0])  # 正方形假设: dy == dx
        return ('BL', (bl_x, bl_y, avg_size))

    elif dy > dx * 1.5:
        # 垂直相邻 → 已知 TL+BL, 推算 TR
        tl = p1 if p1[1] < p2[1] else p2
        bl = p2 if p1[1] < p2[1] else p1
        tr_x = tl[0] + (bl[1] - tl[1])
        tr_y = tl[1]
        return ('TR', (tr_x, tr_y, avg_size))

    else:
        # 对角排列 → TL+BR 或 TR+BL
        if p1[0] < p2[0] and p1[1] < p2[1]:
            tl, br_like = p1, p2
        elif p2[0] < p1[0] and p2[1] < p1[1]:
            tl, br_like = p2, p1
        else:
            # TR + BL 的对角 → 缺失 TL
            if p1[0] > p2[0] and p1[1] < p2[1]:
                tr, bl = p1, p2
            else:
                tr, bl = p2, p1
            tl_x = bl[0]
            tl_y = tr[1]
            return ('TL', (tl_x, tl_y, avg_size))

        # TL + BR 对角 → 推算 TR
        tr_x = br_like[0]
        tr_y = tl[1]
        return ('TR', (tr_x, tr_y, avg_size))


def draw_finder_pattern(img, cx: int, cy: int, module_size: int):
    """在指定位置绘制标准的 7×7 Finder Pattern (含 1 模块宽静区)。

    标准 Finder Pattern 结构 (■=深色模块, □=浅色模块):
      ■■■■■■■
      ■□□□□□■
      ■□■■■□■
      ■□■■■□■
      ■□■■■□■
      ■□□□□□■
      ■■■■■■■

    本函数会先将 (cx, cy) 周围 9x9 模块区域涂白 (含 1 模块静区),
    再按上述模板绘制。深色像素值通过对原图采样估计, 让修复后的
    定位角和原图的整体灰度保持一致, 避免被 WeChatQRCode 因色差排斥。

    Args:
        img:         待修复的 BGR 图像
        cx, cy:      绘制中心坐标
        module_size: 整个 Finder Pattern (7 模块) 的总像素尺寸

    Returns:
        修复后的图像副本 (原图不被修改)。
    """
    result = img.copy()

    # 单模块像素大小
    m = max(1, module_size // 7)
    half = 7 * m // 2

    # 涂白区域: Finder + 1 模块宽静区
    x1 = max(0, cx - half - m)
    y1 = max(0, cy - half - m)
    x2 = min(img.shape[1], cx + half + m)
    y2 = min(img.shape[0], cy + half + m)
    rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # 从原图采样深色像素值, 让修复模块和原图风格匹配
    gray = cvtColor(img, COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    if np_any(gray < 128):
        dark_val = int(np_percentile(gray[gray < 128], 30))
    else:
        dark_val = 0
    dark_color = (dark_val, dark_val, dark_val)

    # 7x7 Finder Pattern 模板: 1=深, 0=浅
    pattern = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]

    origin_x = cx - half
    origin_y = cy - half
    for row in range(7):
        for col in range(7):
            px = origin_x + col * m
            py = origin_y + row * m
            color = dark_color if pattern[row][col] == 1 else (255, 255, 255)
            rectangle(result, (px, py), (px + m - 1, py + m - 1), color, -1)

    return result


def repair_finder_patterns(img):
    """检测 + 推算 + 绘制 三步式 Finder Pattern 修复入口。

    Returns:
        (repaired_img, found_count, repaired_count)
          - repaired_img:    修复后的图像 (无需修复时返回原图)
          - found_count:     原图中检测到的定位角数量 (0~3)
          - repaired_count:  本次修复绘制的定位角数量 (0 或 1)

    决策矩阵:
        found_count >= 3  → 不修复 (定位角齐全)
        found_count <  2  → 不修复 (信息不足以推算)
        found_count == 2  → 推算 + 绘制 1 个缺失角
    """
    corners = detect_finder_patterns(img)
    found_count = len(corners)

    if found_count >= 3:
        return img, found_count, 0

    if found_count < 2:
        logger.debug(
            f"Only {found_count} finder patterns detected, cannot repair."
        )
        return img, found_count, 0

    missing = infer_missing_corner(corners)
    if missing is None:
        return img, found_count, 0

    corner_type, (cx, cy, size) = missing
    logger.debug(
        f"Detected {found_count} corners, missing {corner_type} "
        f"at ({cx},{cy}), size={size}"
    )

    # 越界保护: 推算位置如果落在图像外, 放弃修复
    if cx < 0 or cy < 0 or cx >= img.shape[1] or cy >= img.shape[0]:
        logger.debug(f"Inferred corner ({cx},{cy}) is out of bounds.")
        return img, found_count, 0

    repaired = draw_finder_pattern(img, cx, cy, size)
    return repaired, found_count, 1

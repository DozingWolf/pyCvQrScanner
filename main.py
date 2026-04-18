"""
pyCVQRScanner v2 - 二维码识别 & 文件分发

Usage:
    uv run main.py <输入目录>

功能:
    扫描指定目录下的所有图片, 识别每张图片上的二维码。
    - 识别成功: 以二维码内容作为文件名, 移动到项目根目录的 output/ 目录
    - 识别失败: 保留原文件名, 移动到项目根目录的 failed/ 目录
"""

from pathlib import Path
from re import compile as re_compile
from shutil import move as shutil_move
from sys import argv as sys_argv, exit as sys_exit, stderr as sys_stderr

from loguru import logger

# ── loguru 配置 ───────────────────────────────────────────
logger.remove()
logger.add(
    sys_stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)
logger.add(
    "logs/scan_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
)

from src.pipeline import scan_image

# Windows / NTFS 非法字符 + 控制字符
_ILLEGAL_CHARS = re_compile(r'[<>:"/\\|?*\x00-\x1f]')
# 最大文件名长度 (不含扩展名), 留裕量给序号后缀
_MAX_STEM_LEN = 200


def sanitize_filename(raw: str) -> str:
    """将二维码内容清洗为合法文件名。

    - 替换非法字符为下划线
    - 去除首尾空格和点号 (Windows 不允许文件名以点号结尾)
    - 截断过长文件名
    """
    name = _ILLEGAL_CHARS.sub("_", raw)
    name = name.strip(" .")
    if not name:
        name = "_unnamed_"
    if len(name) > _MAX_STEM_LEN:
        name = name[:_MAX_STEM_LEN]
    return name


def unique_path(dest_dir: Path, stem: str, suffix: str) -> Path:
    """生成不冲突的目标路径, 重名时追加序号 _2, _3 ..."""
    candidate = dest_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        candidate = dest_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def main():
    if len(sys_argv) < 2:
        print("用法: uv run main.py <输入目录>")
        print("示例: uv run main.py ./scanned_images/")
        sys_exit(1)

    input_dir = Path(sys_argv[1])
    if not input_dir.is_dir():
        logger.error(f"输入路径不是有效目录: {input_dir}")
        sys_exit(1)

    # 输出目录与失败目录固定在项目根目录（main.py 所在位置）
    root = Path(__file__).parent
    output_dir = root / "output"
    failed_dir = root / "failed"
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    # 收集图片
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_exts
    )

    if not files:
        logger.warning(f"目录中未发现图片文件: {input_dir}")
        sys_exit(0)

    logger.info(f"共发现 {len(files)} 张图片, 开始识别...")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"失败目录: {failed_dir}")

    success_count = 0
    fail_count = 0

    for src_file in files:
        result = scan_image(src_file)

        if result["status"] == "success" and result["result"]:
            # 识别成功: 用二维码内容重命名, 移至 output/
            clean_name = sanitize_filename(result["result"])
            dest = unique_path(output_dir, clean_name, src_file.suffix.lower())
            shutil_move(str(src_file), str(dest))
            logger.info(
                f"  -> output/{dest.name}  ({result['time_s']:.1f}s)"
            )
            success_count += 1
        else:
            # 识别失败: 保留原名, 移至 failed/
            dest = unique_path(failed_dir, src_file.stem, src_file.suffix.lower())
            shutil_move(str(src_file), str(dest))
            logger.warning(
                f"  -> failed/{dest.name}  ({result['time_s']:.1f}s, {result['status']})"
            )
            fail_count += 1

    # 汇总
    total = success_count + fail_count
    logger.info("=" * 60)
    logger.info(f"处理完成: {total} 张图片")
    logger.info(f"  成功: {success_count}  ({100*success_count/total:.1f}%)")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  输出: {output_dir}")
    logger.info(f"  失败: {failed_dir}")
    logger.info("=" * 60)

    print()
    print(f"处理完成: {success_count}/{total} 成功")
    print(f"  成功文件 -> {output_dir}")
    print(f"  失败文件 -> {failed_dir}")


if __name__ == "__main__":
    main()

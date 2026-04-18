"""
Benchmark script for QR code recognition.

Usage:
    uv run benchmark.py [--scanner <module_path>]

Default scanner: built-in wechatQR baseline (no preprocessing).
Custom scanner: pass a module path exposing a `scan(image_path: str) -> list[str]` function.

Example:
    uv run benchmark.py
    uv run benchmark.py --scanner src.pipeline
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import cv2
from loguru import logger

FAILED_DIR = Path(__file__).parent / "failed"
MODELS_DIR = Path(__file__).parent / "models"
REPORTS_DIR = Path(__file__).parent / "reports"
PER_IMAGE_TIMEOUT = 20.0  # seconds

REPORTS_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add(REPORTS_DIR / "benchmark.log", level="DEBUG", rotation="1 MB", encoding="utf-8")


def baseline_scan(image_path: str) -> list[str]:
    """Baseline scanner using wechatQR with no preprocessing."""
    img = cv2.imread(image_path)
    if img is None:
        return []
    detector = cv2.wechat_qrcode_WeChatQRCode(
        str(MODELS_DIR / "detect.prototxt"),
        str(MODELS_DIR / "detect.caffemodel"),
        str(MODELS_DIR / "sr.prototxt"),
        str(MODELS_DIR / "sr.caffemodel"),
    )
    results, _ = detector.detectAndDecode(img)
    return [r for r in results if r]


def run_benchmark(scan_fn) -> dict:
    images = sorted(FAILED_DIR.glob("*.jpg"))
    if not images:
        logger.error(f"No images found in {FAILED_DIR}")
        sys.exit(1)

    logger.info(f"Starting benchmark on {len(images)} images (timeout={PER_IMAGE_TIMEOUT}s per image)")

    records = []
    for img_path in images:
        start = time.monotonic()
        decoded = []
        timed_out = False
        error_msg = ""

        try:
            # Simple timeout via checking elapsed time isn't feasible in pure Python for C-ext calls.
            # We run in a thread with join timeout instead.
            import threading

            result_holder = []
            exc_holder = []

            def _run():
                try:
                    result_holder.extend(scan_fn(str(img_path)))
                except Exception as e:
                    exc_holder.append(str(e))

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=PER_IMAGE_TIMEOUT)

            elapsed = time.monotonic() - start
            if t.is_alive():
                timed_out = True
                logger.warning(f"[TIMEOUT] {img_path.name} exceeded {PER_IMAGE_TIMEOUT}s")
            elif exc_holder:
                error_msg = exc_holder[0]
                logger.error(f"[ERROR]   {img_path.name}: {error_msg}")
            else:
                decoded = result_holder

        except Exception as e:
            elapsed = time.monotonic() - start
            error_msg = str(e)
            logger.error(f"[ERROR]   {img_path.name}: {error_msg}")

        elapsed = time.monotonic() - start
        success = bool(decoded) and not timed_out

        status_tag = "TIMEOUT" if timed_out else ("OK" if success else "FAIL")
        decoded_str = decoded[0][:40] if decoded else ""
        logger.info(f"[{status_tag:<7}] {img_path.name:<30} {elapsed:6.2f}s  {decoded_str}")

        records.append({
            "name": img_path.name,
            "success": success,
            "timed_out": timed_out,
            "elapsed": elapsed,
            "decoded": decoded,
            "error": error_msg,
        })

    return _summarize(records)


def _summarize(records: list[dict]) -> dict:
    total = len(records)
    success_count = sum(1 for r in records if r["success"])
    timeout_count = sum(1 for r in records if r["timed_out"])
    fail_count = total - success_count
    rate = success_count / total * 100 if total else 0
    avg_time = sum(r["elapsed"] for r in records) / total if total else 0
    max_rec = max(records, key=lambda r: r["elapsed"])

    logger.info("=" * 60)
    logger.info(f"Total: {total}  Success: {success_count}  Fail: {fail_count}  Timeout: {timeout_count}")
    logger.info(f"Recognition rate: {rate:.1f}%")
    logger.info(f"Avg time: {avg_time:.2f}s  Max time: {max_rec['elapsed']:.2f}s ({max_rec['name']})")
    logger.info("=" * 60)

    return {
        "records": records,
        "total": total,
        "success": success_count,
        "fail": fail_count,
        "timeout": timeout_count,
        "rate": rate,
        "avg_time": avg_time,
        "max_time": max_rec["elapsed"],
        "max_name": max_rec["name"],
    }


def write_report(summary: dict, scanner_label: str) -> Path:
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = REPORTS_DIR / "benchmark.md"

    lines = [
        f"# QR Scanner Benchmark Report",
        f"",
        f"**Date**: {now}  ",
        f"**Scanner**: `{scanner_label}`  ",
        f"**Dataset**: `failed/` ({summary['total']} images)  ",
        f"**Timeout per image**: {PER_IMAGE_TIMEOUT}s  ",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total images | {summary['total']} |",
        f"| Recognized | {summary['success']} |",
        f"| Failed | {summary['fail']} |",
        f"| Timed out | {summary['timeout']} |",
        f"| **Recognition rate** | **{summary['rate']:.1f}%** |",
        f"| Avg processing time | {summary['avg_time']:.2f}s |",
        f"| Slowest image | `{summary['max_name']}` ({summary['max_time']:.2f}s) |",
        f"",
        f"## Per-Image Results",
        f"",
        f"| # | Image | Status | Time (s) | Decoded Value |",
        f"|---|-------|--------|----------|---------------|",
    ]

    for i, r in enumerate(summary["records"], 1):
        if r["timed_out"]:
            status = "TIMEOUT"
        elif r["success"]:
            status = "OK"
        else:
            status = "FAIL"

        decoded_preview = r["decoded"][0][:50] if r["decoded"] else (r["error"] or "-")
        decoded_preview = decoded_preview.replace("|", "\\|")
        lines.append(f"| {i} | `{r['name']}` | {status} | {r['elapsed']:.2f} | {decoded_preview} |")

    lines += [
        f"",
        f"## Pass / Fail Decision",
        f"",
    ]

    target_rate = 90.0
    if summary["rate"] >= target_rate:
        lines.append(f"> **PASS** — Recognition rate {summary['rate']:.1f}% >= {target_rate}% target.")
    else:
        lines.append(f"> **FAIL** — Recognition rate {summary['rate']:.1f}% < {target_rate}% target. Further improvement required.")

    content = "\n".join(lines) + "\n"
    report_path.write_text(content, encoding="utf-8")
    logger.info(f"Report written to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="QR Scanner Benchmark")
    parser.add_argument(
        "--scanner",
        default="",
        help="Module path exposing scan(image_path: str) -> list[str]. Default: built-in wechatQR baseline.",
    )
    args = parser.parse_args()

    if args.scanner:
        try:
            mod = importlib.import_module(args.scanner)
            if hasattr(mod, "scan"):
                # 标准接口: scan(image_path: str) -> list[str]
                scan_fn = mod.scan
            elif hasattr(mod, "scan_image"):
                # pipeline.py 的接口: scan_image(path) -> dict
                # 适配为 benchmark 所需的 list[str] 格式
                _scan_image = mod.scan_image
                def scan_fn(image_path: str) -> list[str]:
                    r = _scan_image(image_path)
                    return [r["result"]] if r.get("result") else []
            else:
                raise AttributeError("模块必须暴露 scan() 或 scan_image() 函数")
            scanner_label = args.scanner
            logger.info(f"Using custom scanner: {args.scanner}")
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load scanner '{args.scanner}': {e}")
            sys.exit(1)
    else:
        scan_fn = baseline_scan
        scanner_label = "baseline (wechatQR, no preprocessing)"
        logger.info("Using baseline wechatQR scanner (no preprocessing)")

    summary = run_benchmark(scan_fn)
    report_path = write_report(summary, scanner_label)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()

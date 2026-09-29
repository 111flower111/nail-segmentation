#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_labeling.py —— 生成 labelme 标注工作区
================================================

原始数据(只读, 绝不修改):
    nail_disease_dataset/{train,test}/{healthy,onychomycosis,psoriasis}/*.jpg

产出 (默认 nail_seg/):
    labels.txt                      labelme --labels 文件 (3 个前景类)
    class_map.json                  类别名 -> 掩膜整数索引 / 调色板 (已存在则不覆盖)
    images/train/<cls>/<stem>.jpg   统一后缀、统一方向、统一 RGB
    images/val/<cls>/<stem>.jpg     (原始 test 目录 -> val)
    splits/train.txt                MMSeg split 列表, 每行 "<cls>/<stem>"
    splits/val.txt
    qc/stem_mapping.csv             原始文件名 -> 新文件名 映射(可追溯)
    qc/duplicates.txt               完全重复图 / train-val 泄漏检测
    qc/report.txt                   本次整理汇总

为什么要做图像归一化:
    1) 扩展名混乱(.jpg/.jpeg/.JPG/.jfif/.png/.webp) —— MMSeg 的 CustomDataset 只认单一
       img_suffix, 且个别扩展名会让部分工具链读图失败;
    2) EXIF 方向 —— OpenCV 会按 EXIF 自动旋转, PIL/labelme 不会。二者不一致会导致
       标注好的 mask 与图像 180 度错位。归一化(exif_transpose)后彻底消除该风险;
    3) 文件名里的空格/中文/多个点号 —— 在 split txt、shell、部分数据加载器里都容易出问题,
       统一清洗为 [A-Za-z0-9_-] 并保证唯一。

用法:
    python3 nail_seg/tools/prepare_labeling.py                 # 全量整理
    python3 nail_seg/tools/prepare_labeling.py --force         # 覆盖已存在的工作区
    python3 nail_seg/tools/prepare_labeling.py --link          # 软链接代替拷贝(不做归一化)
    python3 nail_seg/tools/prepare_labeling.py --dry-run       # 只统计不写文件
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

# ---------------------------------------------------------------- 常量
SPLIT_SRC_TO_DST = {"train": "train", "test": "val"}  # 原始 test 作为验证集
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif", ".tif", ".tiff"}
DEFAULT_CLASSES = ["_background_", "healthy", "onychomycosis", "psoriasis"]
DEFAULT_PALETTE = {
    "_background_": [0, 0, 0],
    "healthy": [46, 204, 113],
    "onychomycosis": [241, 196, 15],
    "psoriasis": [231, 76, 60],
}
DEFAULT_CLASS_NAMES_CN = {
    "_background_": "背景(皮肤/指甲以外区域)",
    "healthy": "健康指甲",
    "onychomycosis": "甲癣(灰指甲)",
    "psoriasis": "甲银屑病",
}
# 标注时可以直接输入这些别名(labelme 的标签框里手打中文也认), 统一折算到 classes 里的名字
DEFAULT_LABEL_ALIASES = {
    "background": "_background_", "bg": "_background_", "背景": "_background_",
    "健康": "healthy", "健康指甲": "healthy", "正常": "healthy", "normal": "healthy",
    "甲癣": "onychomycosis", "灰指甲": "onychomycosis", "真菌": "onychomycosis",
    "甲真菌病": "onychomycosis", "fungal": "onychomycosis",
    "银屑病": "psoriasis", "甲银屑病": "psoriasis", "牛皮癣": "psoriasis",
}


# ---------------------------------------------------------------- 工具函数
def sanitize_stem(stem: str, max_len: int = 80) -> str:
    """把文件名主干清洗成 [A-Za-z0-9_-]，避免空格/中文/点号在各工具链里出问题。"""
    # 全角转半角, 去掉变音符号
    stem = "".join(
        c for c in unicodedata.normalize("NFKD", stem)
        if not unicodedata.combining(c)
    )
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)
    stem = re.sub(r"_{2,}", "_", stem).strip("_-")
    if not stem:
        stem = "img"
    if len(stem) > max_len:
        # 超长名字保留尾部(通常尾部的样本编号更有信息量)
        stem = stem[-max_len:].strip("_-")
    return stem


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def normalize_image(src: Path, dst: Path) -> str:
    """
    读图 -> 按 EXIF 摆正 -> RGB -> 写 .jpg。
    若源本身就是无需旋转的 JPEG(RGB)，直接字节拷贝, 避免二次压缩。
    返回处理方式标记, 便于统计。
    """
    with Image.open(src) as im:
        orientation = im.getexif().get(0x0112, 1)
        already_ok = (im.format == "JPEG") and (orientation == 1) and (im.mode == "RGB")
        if already_ok:
            shutil.copy2(src, dst)
            return "copy_jpeg_bytes"
        fixed = ImageOps.exif_transpose(im).convert("RGB")
        fixed.save(dst, "JPEG", quality=95, subsampling=0, optimize=True)
        return "reencode_exif" if orientation != 1 else f"reencode_{im.format}"


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(
        description="整理原始图片, 生成 labelme 标注工作区",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    here = Path(__file__).resolve().parent.parent          # nail_seg/
    ap.add_argument("--src", type=Path, default=here.parent / "nail_disease_dataset",
                    help="原始数据集根目录")
    ap.add_argument("--dst", type=Path, default=here, help="标注工作区根目录(nail_seg)")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES[1:]),
                    help="前景类别名, 逗号分隔, 顺序即掩膜整数 1,2,3...")
    ap.add_argument("--link", action="store_true",
                    help="用软链接代替拷贝(不做图像归一化, 存在 EXIF 错位风险)")
    ap.add_argument("--drop-duplicates", action="store_true",
                    help="内容完全相同的图片只保留一张(train 优先于 val), 避免数据泄漏与重复标注")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    ap.add_argument("--dry-run", action="store_true", help="只统计, 不写任何文件")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        print("[x] --classes 不能为空", file=sys.stderr)
        return 2
    all_classes = ["_background_"] + classes

    if not args.src.is_dir():
        print(f"[x] 原始数据目录不存在: {args.src}", file=sys.stderr)
        return 2

    # ---------- 1. 扫描源数据 ----------
    plan: list[tuple[str, str, Path, str]] = []   # (split, cls, src_path, new_stem)
    used_stems: dict[tuple[str, str], set] = defaultdict(set)
    problems: list[str] = []
    missing_cls: list[str] = []

    for src_split, dst_split in SPLIT_SRC_TO_DST.items():
        sdir = args.src / src_split
        if not sdir.is_dir():
            problems.append(f"缺少划分目录: {sdir}")
            continue
        for cls in classes:
            cdir = sdir / cls
            if not cdir.is_dir():
                missing_cls.append(f"{src_split}/{cls}")
                continue
            files = sorted(p for p in cdir.iterdir()
                           if p.is_file() and p.suffix.lower() in IMG_EXTS)
            for p in files:
                stem = sanitize_stem(p.stem)
                key = (dst_split, cls)
                if stem in used_stems[key]:
                    # 清洗后重名 -> 附加原路径哈希保证唯一
                    stem = f"{stem}_{hashlib.md5(str(p).encode()).hexdigest()[:6]}"
                used_stems[key].add(stem)
                plan.append((dst_split, cls, p, stem))

    if missing_cls:
        print(f"[!] 原始数据中找不到这些类别目录, 已跳过: {', '.join(missing_cls)}", file=sys.stderr)
    if not plan:
        print("[x] 没有扫描到任何图片", file=sys.stderr)
        return 2

    # ---------- 2. 目录与元数据 ----------
    images_root = args.dst / "images"
    splits_dir = args.dst / "splits"
    qc_dir = args.dst / "qc"
    for d in (images_root, splits_dir, qc_dir, args.dst / "masks"):
        d.mkdir(parents=True, exist_ok=True)

    class_map_path = args.dst / "class_map.json"
    labels_path = args.dst / "labels.txt"
    if not args.dry_run:
        if args.force or not class_map_path.exists():
            class_map = {
                "version": 1,
                "task": "nail_disease_semantic_segmentation",
                "description": "指甲病害多类别语义分割。掩膜为单通道 uint8 PNG，像素值即类别索引。",
                "classes": all_classes,
                "class_names_cn": {k: DEFAULT_CLASS_NAMES_CN.get(k, k) for k in all_classes},
                "palette": {k: DEFAULT_PALETTE.get(k, [255, 255, 255]) for k in all_classes},
                "ignore_index": 255,
                "label_aliases": DEFAULT_LABEL_ALIASES,
                "labelme_gui_labels": classes,
                "folder_to_class": {c: c for c in classes},
            }
            class_map_path.write_text(
                json.dumps(class_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        labels_path.write_text("\n".join(classes) + "\n", encoding="utf-8")

    # ---------- 3. 归一化拷贝 ----------
    mapping_rows = []
    digest_kept: dict[str, list[str]] = defaultdict(list)   # 保留的图片按内容分组
    digest_dropped: dict[str, list[str]] = defaultdict(list)  # 去重时丢弃的图片
    method_count: dict[str, int] = defaultdict(int)
    split_lines: dict[str, list[str]] = {s: [] for s in sorted(set(SPLIT_SRC_TO_DST.values()))}
    done = 0
    total = len(plan)

    for dst_split, cls, src_path, stem in plan:
        out_dir = images_root / dst_split / cls
        out_path = out_dir / f"{stem}.jpg"
        rel = f"{dst_split}/{cls}/{stem}.jpg"

        # --- 拷贝/链接 ---
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not args.force:
                method_count["skipped_exists"] += 1
            elif args.link:
                if out_path.exists() or out_path.is_symlink():
                    out_path.unlink()
                os.symlink(src_path.resolve(), out_path)
                method_count["symlink"] += 1
            else:
                method_count[normalize_image(src_path, out_path)] += 1

            # --- 内容去重(作用于归一化后的文件, 因此能识破不同文件名/不同扩展名的同图) ---
            digest = md5_file(out_path)
            prior = digest_kept.get(digest)
            if prior:
                digest_dropped[digest].append(rel)
                if args.drop_duplicates:
                    out_path.unlink()
                    method_count["dropped_duplicate"] += 1
                    mapping_rows.append({
                        "split": dst_split, "class": cls,
                        "original_relative_path": str(src_path.relative_to(args.src)),
                        "new_relative_path": f"{rel}  [DROPPED: 与 {prior[0]} 内容完全相同]",
                    })
                    done += 1
                    if done % 200 == 0 or done == total:
                        print(f"    ... {done}/{total}", flush=True)
                    continue
            digest_kept[digest].append(rel)

        split_lines[dst_split].append(f"{cls}/{stem}")
        mapping_rows.append({
            "split": dst_split, "class": cls,
            "original_relative_path": str(src_path.relative_to(args.src)),
            "new_relative_path": rel,
        })
        done += 1
        if not args.dry_run and (done % 200 == 0 or done == total):
            print(f"    ... {done}/{total}", flush=True)

    # ---------- 4. 写 split 文件 ----------
    if not args.dry_run:
        seen_in_split = {}
        for split, lines in split_lines.items():
            lines = sorted(lines, key=lambda x: (x.split("/")[0], x))
            p = splits_dir / f"{split}.txt"
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            seen_in_split[split] = set(lines)

        with open(qc_dir / "stem_mapping.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(mapping_rows[0].keys()))
            w.writeheader()
            w.writerows(mapping_rows)

    # ---------- 5. 重复图 / 泄漏检测 ----------
    all_groups = {k: digest_kept[k] + digest_dropped.get(k, [])
                  for k in set(digest_kept) | set(digest_dropped)}
    dup_groups = {k: v for k, v in all_groups.items() if len(v) > 1}
    leakage = [v for v in dup_groups.values()
               if {"train", "val"} <= {m.split("/")[0] for m in v}]
    if not args.dry_run:
        with open(qc_dir / "duplicates.txt", "w", encoding="utf-8") as f:
            f.write(f"# 内容完全重复的图片组: {len(dup_groups)} 组"
                    f"{' (已去重, 每组只保留第一张)' if args.drop_duplicates else ' (未去重)'}\n")
            for k, v in sorted(dup_groups.items()):
                f.write(f"{k[:12]}  {len(v)} 张: {', '.join(v)}\n")
            f.write(f"\n# train/val 交叉泄漏组: {len(leakage)} 组"
                    f" (不处理会让验证指标虚高)\n")
            for v in leakage:
                f.write("  " + " | ".join(v) + "\n")

    # ---------- 6. 报告 ----------
    kept_per_cls: dict[tuple[str, str], int] = defaultdict(int)
    for dst_split, lines in split_lines.items():
        for x in lines:
            kept_per_cls[(dst_split, x.split("/")[0])] += 1

    report_lines = [
        "=" * 72,
        "labelme 标注工作区整理报告",
        "=" * 72,
        f"源数据      : {args.src}",
        f"工作区      : {args.dst}",
        f"类别(掩膜)  : " + ", ".join(f"{i}={c}" for i, c in enumerate(all_classes)),
        f"源图片总数  : {total}",
        f"待标注图片  : {sum(len(v) for v in split_lines.values())}"
        + (f" (去重丢弃 {sum(len(v) for v in digest_dropped.values())} 张)" if args.drop_duplicates else ""),
        "",
        "每个划分/类别的待标注图片数:",
    ]
    for split in sorted(split_lines):
        tot = sum(v for (s, _), v in kept_per_cls.items() if s == split)
        report_lines.append(f"  [{split}] 共 {tot}")
        for cls in classes:
            report_lines.append(f"      {cls:16s} {kept_per_cls.get((split, cls), 0)}")
    report_lines += [
        "",
        "图像处理方式统计: " + ", ".join(f"{k}={v}" for k, v in sorted(method_count.items())),
        f"内容重复组      : {len(dup_groups)}  (详见 qc/duplicates.txt)",
        f"train/val 泄漏组: {len(leakage)}" + ("  [已按 train 优先去重]" if args.drop_duplicates else "  [未处理]"),
    ]
    if problems:
        report_lines += ["", "问题:"] + [f"  - {p}" for p in problems]
    report_lines.append("=" * 72)
    report = "\n".join(report_lines)
    print(report)
    if not args.dry_run:
        (qc_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
        print(f"\n[ok] 工作区已就绪: {args.dst}")
        print(f"     下一步: bash {args.dst}/tools/launch_labelme.sh train onychomycosis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

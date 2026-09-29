#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_masks.py —— 掩膜质量校验 + 可视化抽查
=============================================

做四件事:
  1. 逐 划分/类别 统计前景占比、空掩膜、异常尺寸;
  2. 检查掩膜里出现的类别索引是否与所在文件夹类别一致(最常见的标注错误);
  3. 每类随机抽 N 张, 拼成 [原图 | 彩色掩膜 | 叠加] 三联图, 存到 qc/check/ 供肉眼抽查;
  4. 汇总可疑样本清单 qc/check/suspicious.txt。

用法:
    python3 nail_seg/tools/check_masks.py                 # 每类抽 8 张
    python3 nail_seg/tools/check_masks.py --per-class 20  # 每类抽 20 张
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


def find_image(mask_path: Path, img_dir: Path) -> Path | None:
    for e in IMG_EXTS:
        p = img_dir / f"{mask_path.stem}{e}"
        if p.is_file():
            return p
    return None


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="掩膜质量校验")
    ap.add_argument("--root", type=Path, default=here, help="图片/标注所在的根目录")
    ap.add_argument("--out", type=Path, default=None, help="掩膜/QC 的输出根目录(默认与 --root 相同)")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--per-class", type=int, default=8, help="每类抽查张数")
    ap.add_argument("--tile", type=int, default=260, help="三联图单张缩放边长")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cm_path = args.root / "class_map.json"
    if not cm_path.is_file():
        cm_path = args.root / "nail_seg" / "class_map.json"
    cm = json.loads(cm_path.read_text(encoding="utf-8"))
    classes = list(cm["classes"])
    name_to_id = {c: i for i, c in enumerate(classes)}
    palette = {c: tuple(int(v) for v in cm["palette"][c]) for c in classes}
    lut = np.zeros((256, 3), np.uint8)
    for i, c in enumerate(classes):
        r, g, b = palette[c]
        lut[i] = (b, g, r)

    out_root = args.out or args.root
    masks_root = out_root / "masks"
    images_root = args.root / "images"
    out_dir = out_root / "qc" / "check"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    summary: list[str] = []
    suspicious: list[str] = []
    total_masks = 0

    for split in splits:
        sdir = masks_root / split
        if not sdir.is_dir():
            summary.append(f"[!] 没有掩膜目录: {sdir}")
            continue
        for cls_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            cls = cls_dir.name
            exp_id = name_to_id.get(cls)
            files = sorted(cls_dir.glob("*.png"))
            total_masks += len(files)
            ratios, empty, wrong_cls, size_bad = [], 0, [], 0
            for mp in files:
                m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
                if m is None or m.ndim != 2 or m.dtype != np.uint8:
                    suspicious.append(f"[掩膜非法] {split}/{cls}/{mp.name} "
                                      f"(ndim={None if m is None else m.ndim}, dtype={None if m is None else m.dtype})")
                    continue
                ip = find_image(mp, images_root / split / cls)
                if ip is None:
                    suspicious.append(f"[找不到原图] {split}/{cls}/{mp.name}")
                    continue
                img = cv2.imread(str(ip))
                if img is None:
                    suspicious.append(f"[原图读失败] {ip}")
                    continue
                if img.shape[:2] != m.shape[:2]:
                    size_bad += 1
                    suspicious.append(f"[尺寸不一致] {split}/{cls}/{mp.name} "
                                      f"mask={m.shape[:2]} image={img.shape[:2]}")
                    continue
                fg = m > 0
                ratio = float(fg.mean())
                ratios.append(ratio)
                if not fg.any():
                    empty += 1
                    suspicious.append(f"[空掩膜] {split}/{cls}/{mp.name}")
                elif ratio > 0.92:
                    suspicious.append(f"[前景过大 {ratio:.0%}] {split}/{cls}/{mp.name} —— 可能把皮肤一起框了")
                elif ratio < 0.02:
                    suspicious.append(f"[前景过小 {ratio:.1%}] {split}/{cls}/{mp.name} —— 可能漏标/只点了一下")
                ids = set(int(v) for v in np.unique(m)) - {0}
                if exp_id is not None and ids - {exp_id}:
                    wrong_cls.append(mp.name)
                    suspicious.append(
                        f"[类别不符] {split}/{cls}/{mp.name} 掩膜含类别 "
                        f"{sorted(classes[i] for i in ids)} 但位于 {cls}/")

            st = (f"{split}/{cls:<16} 掩膜 {len(files):>5} 张 | "
                  f"前景占比 均值{np.mean(ratios) if ratios else 0:.1%} "
                  f"最小{np.min(ratios) if ratios else 0:.1%} "
                  f"最大{np.max(ratios) if ratios else 0:.1%} | "
                  f"空{empty} 类别错{len(wrong_cls)} 尺寸错{size_bad}")
            summary.append(st)

            # --- 抽查三联图 ---
            picks = rng.sample(files, min(args.per_class, len(files))) if files else []
            rows = []
            for mp in picks:
                ip = find_image(mp, images_root / split / cls)
                if ip is None:
                    continue
                img = cv2.imread(str(ip))
                m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
                if img is None or m is None or img.shape[:2] != m.shape[:2]:
                    continue
                s = args.tile
                img_s = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
                color = lut[m]
                color_s = cv2.resize(color, (s, s), interpolation=cv2.INTER_NEAREST)
                ov = cv2.addWeighted(img, 0.55, color, 0.45, 0)
                ov_s = cv2.resize(ov, (s, s), interpolation=cv2.INTER_AREA)
                # 左上角标注文件名, 方便定位问题样本
                tile = np.hstack([img_s, np.full((s, 4, 3), 255, np.uint8), color_s,
                                  np.full((s, 4, 3), 255, np.uint8), ov_s])
                cv2.putText(tile, mp.stem[:28], (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 1, cv2.LINE_AA)
                rows.append(tile)
            if rows:
                per_row = 2
                while len(rows) % per_row:
                    rows.append(np.zeros_like(rows[0]))
                strip = np.vstack([np.hstack(rows[i:i + per_row]) for i in range(0, len(rows), per_row)])
                cv2.imwrite(str(out_dir / f"{split}_{cls}.jpg"), strip)

    lines = ["=" * 78, "掩膜校验汇总", "=" * 78,
             f"掩膜总数: {total_masks}", ""] + summary + [
             "", f"可疑条目: {len(suspicious)} 条 (详见 qc/check/suspicious.txt)", "=" * 78]
    report = "\n".join(lines)
    print(report)
    (out_dir / "summary.txt").write_text(report + "\n", encoding="utf-8")
    (out_dir / "suspicious.txt").write_text(
        ("\n".join(suspicious) + "\n") if suspicious else "# 未发现可疑掩膜\n", encoding="utf-8")
    print(f"\n抽查三联图: {out_dir}/<split>_<class>.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

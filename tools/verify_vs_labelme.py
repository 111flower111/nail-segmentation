#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_vs_labelme.py —— 用 labelme 官方实现复核本工具产出的掩膜
================================================================

labelme2mask.py 是自己在 PIL 上复刻 labelme 的光栅化逻辑的。为了排除"复刻走样"，
这个脚本直接调用 **已安装的 labelme 自己的 `shapes_to_label()`**，
对同一批 JSON 生成一份参考掩膜，再与本工具的 PNG 做逐像素比对。

必须用装了 labelme 的解释器运行(即 .venv-labelme/bin/python)。

用法:
    .venv-labelme/bin/python nail_seg/tools/verify_vs_labelme.py
    .venv-labelme/bin/python nail_seg/tools/verify_vs_labelme.py --root nail_seg/.selftest
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

try:
    from labelme._utils._image import img_b64_to_arr
    from labelme._utils._shape import shapes_to_label
except ImportError as e:                       # noqa: BLE001
    print(f"[x] 需要在装有 labelme 的解释器下运行: {e}", file=sys.stderr)
    print("    例如: .venv-labelme/bin/python nail_seg/tools/verify_vs_labelme.py", file=sys.stderr)
    raise SystemExit(2)

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="与 labelme 官方实现比对掩膜")
    ap.add_argument("--root", type=Path, default=here)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--limit", type=int, default=0, help="每个划分最多比对多少张(0=全部)")
    args = ap.parse_args()

    cm = json.loads((args.root / "class_map.json").read_text(encoding="utf-8"))
    classes = list(cm["classes"])
    # labelme 的 shapes_to_label 只认识前景类, 背景(0)由全零数组表示
    name_to_value = {c: i for i, c in enumerate(classes) if c != "_background_"}
    # 本工具支持中文别名, labelme 不支持; 比对前先把别名折算成规范类名, 否则会被误判为不一致
    alias = dict(cm.get("label_aliases", {}))
    for en, cn in cm.get("class_names_cn", {}).items():
        alias.setdefault(cn, en)

    def canon(label: str) -> str:
        return alias.get(label.strip(), label.strip())

    n_ok = n_bad = n_skip = n_na = 0
    checked = 0
    bad_list: list[str] = []
    na_list: list[str] = []

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        sdir = args.root / "images" / split
        if not sdir.is_dir():
            continue
        for cls_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            done_here = 0
            for jp in sorted(cls_dir.glob("*.json")):
                if args.limit and done_here >= args.limit:
                    break
                mp = args.root / "masks" / split / cls_dir.name / f"{jp.stem}.png"
                if not mp.is_file():
                    n_skip += 1
                    continue
                meta = json.loads(jp.read_text(encoding="utf-8"))
                h = meta.get("imageHeight")
                w = meta.get("imageWidth")
                if not h or not w:                       # 没写尺寸就从原图取
                    import PIL.Image
                    img = next((cls_dir / f"{jp.stem}{e}" for e in IMG_EXTS
                                if (cls_dir / f"{jp.stem}{e}").is_file()), None)
                    if img is None:
                        n_skip += 1
                        continue
                    with PIL.Image.open(img) as im:
                        w, h = im.size

                shapes = []
                for s in meta.get("shapes", []):
                    sh = {"label": canon(s["label"]), "points": s["points"],
                          "shape_type": s.get("shape_type") or "polygon",
                          "group_id": s.get("group_id")}
                    if sh["shape_type"] == "mask" and s.get("mask"):
                        sh["mask"] = img_b64_to_arr(s["mask"]).astype(bool)
                    else:
                        sh["mask"] = None
                    shapes.append(sh)
                if any(sh["label"] == "_background_" for sh in shapes):
                    # 本工具允许显式画 _background_ 多边形来挖洞, labelme 没有"画背景"的概念,
                    # 这类文件无法做 1:1 比对, 归为"不适用"而不是"不一致"
                    n_na += 1
                    na_list.append(f"{split}/{cls_dir.name}/{jp.stem}: 含显式 _background_ 多边形")
                    continue
                try:
                    ref, _ = shapes_to_label(img_shape=(h, w), shapes=shapes,
                                             label_name_to_value=name_to_value)
                except Exception as e:                    # noqa: BLE001
                    bad_list.append(f"[labelme 解析失败] {split}/{cls_dir.name}/{jp.name}: {e}")
                    n_bad += 1
                    continue

                import PIL.Image
                with PIL.Image.open(mp) as m:
                    got = np.array(m)
                checked += 1
                done_here += 1
                if got.shape != ref.shape:
                    n_bad += 1
                    bad_list.append(f"[尺寸不一致] {split}/{cls_dir.name}/{jp.stem}: "
                                    f"本工具{got.shape} vs labelme{ref.shape}")
                    continue
                diff = int((got.astype(np.int32) != ref.astype(np.int32)).sum())
                if diff == 0:
                    n_ok += 1
                else:
                    n_bad += 1
                    bad_list.append(f"[像素不一致] {split}/{cls_dir.name}/{jp.stem}: {diff} 个像素不同")

    lines = ["=" * 74, "掩膜 vs labelme 官方实现 逐像素比对", "=" * 74,
             f"labelme 版本: {__import__('labelme').__version__ if hasattr(__import__('labelme'), '__version__') else '?'}",
             f"工作区      : {args.root}",
             f"比对 {checked} 张 | 完全一致 {n_ok} | 不一致 {n_bad} | "
             f"不适用 {n_na} | 跳过(无掩膜) {n_skip}", "=" * 74]
    report = "\n".join(lines)
    print(report)
    if bad_list:
        print("\n不一致明细:")
        for x in bad_list[:40]:
            print("  -", x)
    if na_list:
        print("\n不适用(labelme 无对应语义):")
        for x in na_list[:10]:
            print("  -", x)
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

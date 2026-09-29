#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask2color.py —— 把整数掩膜渲染成人眼可看的彩色图
====================================================

为什么需要它:
    语义分割的整数掩膜是 8 位灰度 PNG, 像素值 = 类别索引(0/1/2/3)。
    前景强度只有 1~3 (满量程 255), 用任何看图软件打开都是**黑的**,
    和背景(0)肉眼分不出 —— 文件本身是对的, 只是不可视化。
    训练时必须用 masks/ 里的整数掩膜; masks_color/ 只是给人看的, 不要拿去训练。

产出:
    <out>/masks_color/<split>/<cls>/*.png     RGB 彩色掩膜(可直接双击打开)
    <out>/qc_color/contact_<split>_<cls>.jpg  该类别**全部**图片的缩略拼图(叠加图, 带图例)

用法:
    python3 mask2color.py --root . --out .
    python3 mask2color.py --root . --out DIR --thumb 150 --cols 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


def find_image(stem: str, img_dir: Path) -> Path | None:
    for e in IMG_EXTS:
        p = img_dir / f"{stem}{e}"
        if p.is_file():
            return p
    return None


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="整数掩膜 -> 彩色可视化",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="图片所在根目录(需含 images/)")
    ap.add_argument("--out", type=Path, default=None, help="掩膜/输出根目录(默认同 --root)")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--thumb", type=int, default=170, help="拼图缩略图边长")
    ap.add_argument("--cols", type=int, default=7, help="拼图每行张数")
    ap.add_argument("--no-contact", action="store_true", help="不生成缩略拼图")
    args = ap.parse_args()

    out_root = args.out or args.root
    cm_path = args.root / "class_map.json"
    if not cm_path.is_file():
        cm_path = args.root / "nail_seg" / "class_map.json"
    cm = json.loads(cm_path.read_text(encoding="utf-8"))
    classes = list(cm["classes"])
    palette = {c: tuple(int(v) for v in cm["palette"][c]) for c in classes}

    lut = np.zeros((256, 3), np.uint8)          # RGB
    for i, c in enumerate(classes):
        lut[i] = palette[c]

    masks_root = out_root / "masks"
    images_root = args.root / "images"
    color_root = out_root / "masks_color"
    qc_root = out_root / "qc_color"
    qc_root.mkdir(parents=True, exist_ok=True)

    n_total = 0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        sdir = masks_root / split
        if not sdir.is_dir():
            print(f"[!] 没有掩膜目录: {sdir}")
            continue
        for cls_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            cls = cls_dir.name
            files = sorted(cls_dir.glob("*.png"))
            thumbs = []
            for mp in files:
                m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
                if m is None or m.ndim != 2:
                    print(f"[!] 跳过非单通道掩膜: {mp}")
                    continue
                rgb = lut[m]                                  # 查表上色
                dst = color_root / split / cls / mp.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb, "RGB").save(dst)         # 必须用 RGB 写, 否则颜色会串
                n_total += 1

                if not args.no_contact:
                    ip = find_image(mp.stem, images_root / split / cls)
                    if ip is None:
                        continue
                    img = cv2.imread(str(ip))
                    if img is None or img.shape[:2] != m.shape[:2]:
                        continue
                    ov = cv2.addWeighted(img, 0.55, rgb[:, :, ::-1], 0.45, 0)
                    t = args.thumb
                    ov = cv2.resize(ov, (t, t), interpolation=cv2.INTER_AREA)
                    cv2.putText(ov, mp.stem[:22], (3, 12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.34, (255, 255, 255), 1, cv2.LINE_AA)
                    thumbs.append(ov)

            if thumbs:
                cols = args.cols
                rows = (len(thumbs) + cols - 1) // cols
                t = args.thumb
                sheet = np.full((rows * t + 34, cols * t, 3), 24, np.uint8)
                for i, th in enumerate(thumbs):
                    r, c = divmod(i, cols)
                    sheet[34 + r * t:34 + r * t + th.shape[0], c * t:c * t + th.shape[1]] = th
                # 顶部图例: 类别 + 该类别颜色
                x = 6
                cv2.putText(sheet, f"{split}/{cls}", (x, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
                x += 210
                for c in classes:
                    r, g, b = palette[c]
                    cv2.rectangle(sheet, (x, 8), (x + 16, 24), (b, g, r), -1)
                    cv2.putText(sheet, c, (x + 22, 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (230, 230, 230), 1, cv2.LINE_AA)
                    x += 30 + 9 * len(c)
                out = qc_root / f"contact_{split}_{cls}.jpg"
                cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                print(f"  拼图 {len(thumbs):>3} 张 -> {out}")

    print(f"\n[ok] 彩色掩膜 {n_total} 张 -> {color_root}")
    print(f"     缩略拼图 -> {qc_root}/contact_*.jpg")
    print("     注意: 训练请用 masks/ 里的整数掩膜, masks_color/ 仅供肉眼查看")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labelme2mask.py —— labelme JSON 标注 -> 整数掩膜 (integer mask)
================================================================

输入(由 prepare_labeling.py 生成, 标注时 labelme 把 json 存在图片旁边):
    nail_seg/images/train/onychomycosis/xxx.jpg
    nail_seg/images/train/onychomycosis/xxx.json      <- labelme 手工标注

输出:
    nail_seg/masks/train/onychomycosis/xxx.png        单通道 uint8, 像素值 = 类别索引
    nail_seg/qc/missing_annotations.txt               还没标的图
    nail_seg/qc/class_mismatch.txt                    多边形标签与所属文件夹类别不一致
    nail_seg/qc/convert_report.txt                    汇总
    nail_seg/qc/overlay/...                           叠加可视化(带 --vis)

类别索引来自 class_map.json:
    0=_background_  1=healthy  2=onychomycosis  3=psoriasis
即 MMSeg 里 num_classes=4, classes=('background','healthy','onychomycosis','psoriasis')。

支持的 shape_type: polygon(主力) / rectangle / circle / line / linestrip / point /
                   oriented_rectangle / mask(labelme7 的 SAM 画笔)。

★ 关于「线段」工具:
  labelme 的 linestrip/line 在官方实现里只会画出 line_width(默认 10) 像素宽的**描边**,
  沿指甲描一圈并不会得到"填充的指甲区域"。手工标注时很容易误用这个工具, 这类形状的
  首尾间隙通常只有 1~5 像素(绕一圈回到起点附近), 本质就是闭合轮廓。
  默认 --linestrip auto: 首尾间隙 <= 25% 周长 -> 自动闭合后按多边形填充;
  真正开放的线段 -> 跳过并写入 qc/linestrip_review.txt 供人工复核(不猜)。
  --linestrip stroke 则完全按 labelme 原样处理。

重叠策略: 后画的多边形覆盖先画的 (与 labelme 官方 labelme2voc.py 一致)。

用法:
    python3 nail_seg/tools/labelme2mask.py
    python3 nail_seg/tools/labelme2mask.py --class-source folder   # 类别完全按文件夹定
    python3 nail_seg/tools/labelme2mask.py --vis                   # 同时输出叠加图
    python3 nail_seg/tools/labelme2mask.py --root /path/数据 --out /path/输出
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]

# 允许标注时直接写中文/别名, 统一折算到 class_map.json 里的英文类名
EXTRA_ALIASES = {
    "background": "_background_", "bg": "_background_",
    "背景": "_background_",
    "健康": "healthy", "健康指甲": "healthy", "正常": "healthy", "normal": "healthy",
    "甲癣": "onychomycosis", "灰指甲": "onychomycosis", "真菌": "onychomycosis",
    "甲真菌病": "onychomycosis", "fungal": "onychomycosis",
    "银屑病": "psoriasis", "甲银屑病": "psoriasis", "牛皮癣": "psoriasis",
}


# ---------------------------------------------------------------- 类别表
class ClassTable:
    def __init__(self, class_map_path: Path):
        cm = json.loads(class_map_path.read_text(encoding="utf-8"))
        self.classes: list[str] = list(cm["classes"])
        self.name_to_id = {c: i for i, c in enumerate(self.classes)}
        self.palette = {}
        for c in self.classes:
            rgb = cm.get("palette", {}).get(c, [255, 255, 255])
            self.palette[c] = tuple(int(v) for v in rgb)
        # 中文名/别名 -> 英文名。别名表以 class_map.json 为准(两个脚本共用同一份定义)
        self.alias = dict(EXTRA_ALIASES)
        self.alias.update(cm.get("label_aliases", {}))
        for en, cn in cm.get("class_names_cn", {}).items():
            self.alias.setdefault(cn, en)
        self.ignore_index = int(cm.get("ignore_index", 255))

    def resolve(self, label: str) -> int | None:
        label = (label or "").strip()
        if label in self.name_to_id:
            return self.name_to_id[label]
        if label in self.alias:
            return self.name_to_id.get(self.alias[label])
        return None

    def lut_bgr(self) -> np.ndarray:
        lut = np.zeros((256, 3), np.uint8)
        for i, c in enumerate(self.classes):
            r, g, b = self.palette[c]
            lut[i] = (b, g, r)  # OpenCV 是 BGR
        return lut


# ---------------------------------------------------------------- 几何 -> 掩膜
def _pts(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(-1, 2)


def shape_region(shape: dict, h: int, w: int,
                 line_width: int = 10, point_size: int = 5) -> np.ndarray | None:
    """
    把单个 labelme shape 光栅化成 bool 区域。

    实现与已安装的 labelme 7.7.0 `labelme/_utils/_shape.py::shape_to_mask` 逐行对齐,
    因此本工具产出的掩膜与 labelme 官方转换结果一致:
        polygon            -> ImageDraw.polygon(outline=1, fill=1)       (主力)
        rectangle          -> ImageDraw.rectangle(min/max, outline=1, fill=1)  含两端像素
        circle             -> ImageDraw.ellipse(outline=1, fill=1)
        line               -> ImageDraw.line(width=line_width)
        linestrip          -> ImageDraw.line(width=line_width, joint="curve")
        point              -> 半径=point_size 的圆点
        oriented_rectangle -> 按 4 点多边形填充
        mask               -> 内嵌 base64 PNG 补丁(labelme 7 的 SAM 画笔/AI 辅助标注), 按 bbox 贴回
    返回 None 表示点数不足/类型不支持。
    """
    st = shape.get("shape_type") or "polygon"
    pts = _pts(shape.get("points", []))

    # --- mask: 不是几何图形，而是内嵌的 base64 PNG 补丁(labeleme 7 的 AI 画笔) ---
    if st == "mask":
        b64 = shape.get("mask")
        if not b64:
            return None
        raw = base64.b64decode(b64)
        with Image.open(io.BytesIO(raw)) as patch_im:
            patch = np.array(patch_im).astype(bool)
        if pts.shape[0] < 2:
            return None
        (x1, y1), (x2, y2) = pts.round().astype(int)
        if np.array_equal(pts, np.trunc(pts)):          # 整数 bbox -> 沿用旧版裁剪尺寸
            ph, pw = y2 - y1 + 1, x2 - x1 + 1
        else:
            ph, pw = patch.shape
        y_start, y_stop = max(y1, 0), min(y1 + ph, h)
        x_start, x_stop = max(x1, 0), min(x1 + pw, w)
        region = np.zeros((h, w), dtype=bool)
        if y_start < y_stop and x_start < x_stop:
            region[y_start:y_stop, x_start:x_stop] = patch[
                y_start - y1:y_stop - y1, x_start - x1:x_stop - x1]
        return region

    if pts.shape[0] == 0:
        return None
    xy = [(float(p[0]), float(p[1])) for p in pts]

    # labelme 的掩膜是 uint8 L 模式, 这里保持一致
    img = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
    draw = ImageDraw.Draw(img)

    if st == "polygon":
        if len(xy) < 3:
            return None
        draw.polygon(xy, outline=1, fill=1)
    elif st == "oriented_rectangle":
        if len(xy) != 4:
            return None
        draw.polygon(xy, outline=1, fill=1)
    elif st == "rectangle":
        if len(xy) != 2:
            return None
        (x0, y0), (x1, y1) = xy
        draw.rectangle(((min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))),
                       outline=1, fill=1)
    elif st == "circle":
        if len(xy) != 2:
            return None
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse(((cx - d, cy - d), (cx + d, cy + d)), outline=1, fill=1)
    elif st == "line":
        if len(xy) != 2:
            return None
        draw.line(xy, fill=1, width=line_width)
    elif st == "linestrip":
        if len(xy) < 2:
            return None
        # joint="curve" 让宽线在拐角处不缺口(与 labelme 一致)
        draw.line(xy, fill=1, width=line_width, joint="curve")
    elif st == "point":
        (cx, cy), = xy[:1]
        r = point_size
        draw.ellipse(((cx - r, cy - r), (cx + r, cy + r)), outline=1, fill=1)
    else:
        return None
    return np.array(img, dtype=np.uint8).astype(bool)


def normalize_shape(shape: dict, linestrip_mode: str = "auto", close_ratio: float = 0.25):
    """
    把 labelme 的「线段」类 shape 归一化成可填充的区域。

    返回 (new_shape, note):
        note == ""      -> 原样使用
        note 非空        -> 已归一化(自动闭合填充), note 为说明
        new_shape None  -> 应跳过(开放/退化线段), note 为原因
    """
    st = shape.get("shape_type") or "polygon"
    if st not in ("linestrip", "line"):
        return shape, ""
    if linestrip_mode == "stroke":                 # 完全按 labelme 原样(10px 描边)
        return shape, ""
    pts = _pts(shape.get("points", []))
    if st == "line" or pts.shape[0] < 3:
        return None, (f"{st} 只有 {pts.shape[0]} 个点, 是开放/退化线段, 不构成区域 -> 跳过")
    gap = float(np.linalg.norm(pts[0] - pts[-1]))
    perim = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    ratio = gap / max(perim, 1e-9)
    if ratio <= close_ratio or linestrip_mode == "close":
        poly = dict(shape)
        poly["shape_type"] = "polygon"
        return poly, (f"linestrip 自动闭合后按多边形填充 (首尾间隙 {gap:.1f}px = {ratio:.1%} 周长)")
    return None, (f"开放 linestrip (首尾间隙 {gap:.1f}px = {ratio:.1%} 周长) 不是闭合轮廓 "
                  f"-> 跳过, 请人工复核")


# ---------------------------------------------------------------- 单文件转换
_DIR_INDEX: dict[Path, dict[str, Path]] = {}


def _index_dir(d: Path) -> dict[str, Path]:
    """目录内图片索引: 主干名(小写) 和 完整文件名(小写) 都能查到, 大小写不敏感。"""
    idx = _DIR_INDEX.get(d)
    if idx is None:
        idx = {}
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                idx.setdefault(p.stem.lower(), p)
                idx.setdefault(p.name.lower(), p)
        _DIR_INDEX[d] = idx
    return idx


def find_image(json_path: Path, meta: dict) -> Path | None:
    """定位 json 对应的原图: 先按同名主干, 再按 json 里记录的 imagePath 文件名。"""
    idx = _index_dir(json_path.parent)
    keys = [json_path.stem.lower()]
    ip = meta.get("imagePath")
    if ip:
        base = Path(str(ip).replace("\\", "/")).name
        if base:
            keys.append(base.lower())
            keys.append(Path(base).stem.lower())
    for k in keys:
        if k in idx:
            return idx[k]
    return None


def convert_one(json_path: Path, cls_folder: str, out_path: Path, table: ClassTable,
                class_source: str, unknown: str, warn,
                line_width: int = 10, point_size: int = 5,
                linestrip_mode: str = "auto", close_ratio: float = 0.25) -> dict:
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    img_path = find_image(json_path, meta)
    if img_path is None:
        if meta.get("imageData"):
            import base64, io
            from PIL import Image
            buf = base64.b64decode(meta["imageData"])
            with Image.open(io.BytesIO(buf)) as im:
                w, h = im.size
            img_path = None
        else:
            raise FileNotFoundError(f"找不到对应图片 (json 里也没有 imageData): {json_path}")
    else:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"读图失败: {img_path}")
        h, w = img.shape[:2]

    jh, jw = meta.get("imageHeight"), meta.get("imageWidth")
    if jh and jw and (jh, jw) != (h, w):
        warn(f"[尺寸不一致] {json_path.name}: json({jw}x{jh}) vs 图片({w}x{h}) —— 检查 EXIF/图片被替换")

    folder_cls = cls_folder if cls_folder in table.name_to_id else None
    mask = np.zeros((h, w), np.uint8)
    stats = {"shapes": 0, "drawn": 0, "overlaps": 0, "overlap_px": 0,
             "unknown_labels": [], "empty": True,
             "autoclosed": 0, "autoclose_notes": [], "skipped": []}
    fg = np.zeros((h, w), bool)      # 已画过的前景区域(不含背景类)

    for sh in meta.get("shapes", []):
        label = sh.get("label", "")
        stats["shapes"] += 1
        lid = table.resolve(label)

        if class_source == "folder":
            cid = table.name_to_id.get(folder_cls) if folder_cls else None
        elif class_source == "auto":
            cid = lid if lid is not None else (table.name_to_id.get(folder_cls) if folder_cls else None)
        else:  # label: 完全由多边形自己的标签决定类别
            cid = lid

        if cid is None:
            stats["unknown_labels"].append(label)
            if unknown == "error":
                raise ValueError(
                    f"无法识别的标签 {label!r}; 合法标签={table.classes} "
                    f"(或用 --class-source folder / --unknown ignore)"
                )
            continue

        if class_source != "folder" and lid is not None and folder_cls and \
                table.classes[lid] != folder_cls:
            warn(f"[类别与文件夹不符] {json_path.name}: 多边形标签={table.classes[lid]} 但位于 {cls_folder}/")

        sh2, note = normalize_shape(sh, linestrip_mode, close_ratio)
        if sh2 is None:
            stats["skipped"].append(f"{json_path.name} [{label}] {note}")
            continue
        if note:
            stats["autoclosed"] += 1
            stats["autoclose_notes"].append(f"{json_path.name} [{label}] {note}")

        region = shape_region(sh2, h, w, line_width=line_width, point_size=point_size)
        if region is None:
            stats["skipped"].append(f"{json_path.name} [{label}] 无法解析 "
                                    f"{sh.get('shape_type')} ({len(sh.get('points', []))} 个点)")
            continue

        ov = int(np.count_nonzero(region & fg))
        if ov:
            stats["overlaps"] += 1
            stats["overlap_px"] += ov
        mask[region] = cid          # 后画的覆盖先画的(与 labelme2voc 一致)
        stats["drawn"] += 1
        if cid != 0:
            fg |= region

    stats["empty"] = not fg.any()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), mask):
        raise IOError(f"掩膜写入失败: {out_path}")
    stats["fg_ratio"] = float(fg.mean())
    stats["classes_present"] = sorted(int(v) for v in np.unique(mask))
    return stats


# ---------------------------------------------------------------- 主流程
def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="labelme json -> 整数掩膜",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", type=Path, default=here, help="nail_seg 根目录")
    ap.add_argument("--splits", default="train,val", help="要转换的划分")
    ap.add_argument("--class-map", type=Path, default=None, help="默认 <root>/class_map.json")
    ap.add_argument("--class-source", choices=["label", "folder", "auto"], default="label",
                    help="掩膜类别取多边形标签 / 所属文件夹 / 标签优先失败回退文件夹")
    ap.add_argument("--unknown", choices=["error", "ignore"], default="error",
                    help="遇到 class_map 里没有的标签时: 报错 / 跳过")
    ap.add_argument("--out", type=Path, default=None,
                    help="掩膜/QC 的输出根目录(默认与 --root 相同)")
    ap.add_argument("--linestrip", choices=["auto", "close", "stroke"], default="auto",
                    help="线段类形状的处理: auto=闭合则填充(推荐) / close=一律闭合填充 / "
                         "stroke=完全按 labelme 的10px描边")
    ap.add_argument("--close-ratio", type=float, default=0.25,
                    help="auto 模式下判定为闭合的阈值: 首尾间隙/周长 <= 该值")
    ap.add_argument("--line-width", type=int, default=10,
                    help="linestrip/line 的线宽(与 labelme 默认一致)")
    ap.add_argument("--point-size", type=int, default=5,
                    help="point 的半径(与 labelme 默认一致)")
    ap.add_argument("--vis", action="store_true", help="额外输出 原图|彩色掩膜|叠加 三者拼接图")
    ap.add_argument("--vis-max", type=int, default=0, help="可视化最多输出张数, 0=全部")
    args = ap.parse_args()

    table = ClassTable(args.class_map or (args.root / "class_map.json"))
    out_root = args.out or args.root
    images_root = args.root / "images"
    masks_root = out_root / "masks"
    qc_dir = out_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    warnings: list[str] = []
    warn = warnings.append
    mismatch_lines: list[str] = []
    missing_lines: list[str] = []
    autoclose_notes: list[str] = []
    skipped_notes: list[str] = []
    per_group: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"total": 0, "done": 0, "drawn": 0, "empty": 0, "fg": [], "overlap": 0,
                 "autoclosed": 0})
    vis_budget = args.vis_max if args.vis_max > 0 else 10 ** 9
    vis_done = 0
    errors = 0
    lut = table.lut_bgr()

    for split in splits:
        sdir = images_root / split
        if not sdir.is_dir():
            warn(f"缺少划分目录: {sdir}")
            continue
        for cls_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            cls = cls_dir.name
            for json_path in sorted(cls_dir.glob("*.json")):
                g = per_group[(split, cls)]
                g["total"] += 1
                out_path = masks_root / split / cls / f"{json_path.stem}.png"
                try:
                    st = convert_one(json_path, cls, out_path, table,
                                     args.class_source, args.unknown, warn,
                                     line_width=args.line_width, point_size=args.point_size,
                                     linestrip_mode=args.linestrip, close_ratio=args.close_ratio)
                except Exception as e:                      # noqa: BLE001
                    errors += 1
                    warn(f"[失败] {json_path}: {e}")
                    continue
                g["done"] += 1
                g["drawn"] += st["drawn"]
                g["overlap"] += st["overlaps"]
                g["autoclosed"] += st["autoclosed"]
                autoclose_notes += st["autoclose_notes"]
                skipped_notes += st["skipped"]
                g["fg"].append(st["fg_ratio"])
                if st["empty"]:
                    g["empty"] += 1
                    warn(f"[空掩膜] {split}/{cls}/{json_path.stem}: 该图没有任何前景像素")
                if args.vis and vis_done < vis_budget:
                    img_path = find_image(json_path, json.loads(json_path.read_text(encoding="utf-8")))
                    if img_path:
                        img = cv2.imread(str(img_path))
                        mm = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)
                        color = lut[mm]
                        ov = cv2.addWeighted(img, 0.55, color, 0.45, 0)
                        bar = np.full((img.shape[0], 8, 3), 255, np.uint8)
                        canvas = np.hstack([img, bar, color, bar, ov])
                        vp = qc_dir / "overlay" / split / cls / f"{json_path.stem}.jpg"
                        vp.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(vp), canvas)
                        vis_done += 1

    # 统计每个文件夹里"还没标注"的图片
    total_imgs = 0
    for split in splits:
        sdir = images_root / split
        if not sdir.is_dir():
            continue
        for cls_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            jsons = {p.stem for p in cls_dir.glob("*.json")}
            imgs = [p for p in sorted(cls_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in IMG_EXTS]
            miss = [p.name for p in imgs if p.stem not in jsons]
            total_imgs += len(imgs)
            if miss:
                missing_lines.append(f"# {split}/{cls}: 缺 {len(miss)}/{len(imgs)}")
                missing_lines += [f"  {m}" for m in miss]

    # 类别不符的告警单独落盘
    mismatch_lines = [w for w in warnings if "类别与文件夹不符" in w]

    lines = ["=" * 76, "labelme JSON -> 整数掩膜 转换报告", "=" * 76,
             f"工作区     : {args.root}",
             f"类别映射   : " + ", ".join(f"{i}={c}" for i, c in enumerate(table.classes)),
             f"类别来源   : {args.class_source}",
             f"输出目录   : {out_root}",
             f"线段处理   : {args.linestrip} (闭合阈值 {args.close_ratio:.0%} 周长)", ""]
    tot_done = sum(g["done"] for g in per_group.values())
    tot_total = sum(g["total"] for g in per_group.values())
    lines.append(f"已标注 JSON: {tot_total} 张, 成功转换 {tot_done} 张, 失败 {errors} 张")
    lines.append(f"图片总数    : {total_imgs} 张 (未标注 {total_imgs - tot_done} 张)")
    n_auto = sum(g["autoclosed"] for g in per_group.values())
    if n_auto or skipped_notes:
        lines.append(f"线段自动闭合: {n_auto} 个形状   |   跳过待复核: {len(skipped_notes)} 个形状")
    lines.append("")
    lines.append(f"{'划分/类别':<34}{'已转换':>8}{'多边形':>8}{'空掩膜':>8}{'平均前景占比':>14}")
    for (split, cls), g in sorted(per_group.items()):
        fg = float(np.mean(g["fg"])) if g["fg"] else 0.0
        lines.append(f"{split + '/' + cls:<34}{g['done']:>8}{g['drawn']:>8}{g['empty']:>8}{fg:>13.1%}")
    if warnings:
        lines += ["", f"告警 {len(warnings)} 条 (最多显示 25 条):"]
        lines += [f"  - {w}" for w in warnings[:25]]
    lines.append("=" * 76)
    report = "\n".join(lines)
    print(report)
    (qc_dir / "convert_report.txt").write_text(report + "\n", encoding="utf-8")
    if missing_lines:
        (qc_dir / "missing_annotations.txt").write_text("\n".join(missing_lines) + "\n", encoding="utf-8")
    else:
        (qc_dir / "missing_annotations.txt").write_text("# 全部图片均已标注\n", encoding="utf-8")
    (qc_dir / "class_mismatch.txt").write_text(
        ("\n".join(mismatch_lines) + "\n") if mismatch_lines else "# 无类别不符\n", encoding="utf-8")
    with open(qc_dir / "linestrip_review.txt", "w", encoding="utf-8") as f:
        f.write(f"# 线段类形状处理明细 (--linestrip {args.linestrip})\n")
        f.write(f"\n## 已自动闭合并按多边形填充: {len(autoclose_notes)} 个\n")
        f.writelines(f"  {x}\n" for x in autoclose_notes)
        f.write(f"\n## 跳过待人工复核: {len(skipped_notes)} 个\n")
        f.writelines(f"  {x}\n" for x in skipped_notes)
    print(f"\n掩膜输出: {masks_root}")
    print(f"未标注清单: {qc_dir / 'missing_annotations.txt'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selftest_pipeline.py —— 端到端自测: 合成 labelme JSON -> 整数掩膜 -> 独立校验
==============================================================================

不需要人工标注也能验证整条链路。它会在 nail_seg/.selftest/ 下造一个小型沙盒:
  * 从真实图片里各取一张放到沙盒的 images/train/<cls>/
  * 程序化生成 labelme 格式的 JSON(多边形 / 矩形 / 圆 / 点 / 线段 / 背景覆盖)
  * 调用 labelme2mask.py 转换
  * 用**独立的**光栅化实现(PIL ImageDraw)和解析几何面积做交叉验证
  * 输出三联叠加图 nail_seg/qc/selftest_overlay.jpg 供肉眼确认

用法:
    python3 nail_seg/tools/selftest_pipeline.py
退出码 0 = 全部通过。
"""
from __future__ import annotations

import base64
import io
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent.parent          # nail_seg/
TOOLS = HERE / "tools"
SCRATCH = HERE / ".selftest"
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    RESULTS.append((bool(ok), msg))
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")


def make_json(img_path: Path, shapes: list[dict]) -> dict:
    with Image.open(img_path) as im:
        w, h = im.size
    return {
        "version": "5.5.0", "flags": {}, "shapes": shapes,
        "imagePath": img_path.name, "imageData": None,
        "imageHeight": h, "imageWidth": w,
    }


def shape(label, points, stype="polygon"):
    return {"label": label, "points": [[float(x), float(y)] for x, y in points],
            "group_id": None, "description": "", "shape_type": stype, "flags": {}}


def ellipse_poly(cx, cy, rx, ry, n=72):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1)


def seg_dist(xy: np.ndarray, a, b) -> np.ndarray:
    """点集 xy(N,2) 到线段 ab 的距离(向量化)。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    t = ((xy - a) @ ab) / max(L2, 1e-9)
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(xy - proj, axis=1)


def main() -> int:
    if not (HERE / "class_map.json").is_file():
        print("[x] 先运行 prepare_labeling.py 生成工作区", file=sys.stderr)
        return 2
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)

    print("=" * 72)
    print("端到端自测: labelme JSON -> 整数掩膜")
    print("=" * 72)

    # ---------------- 1. 造沙盒 ----------------
    print("\n[1/5] 构造沙盒数据 ...")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "class_map.json", SCRATCH / "class_map.json")
    shutil.copy2(HERE / "labels.txt", SCRATCH / "labels.txt")
    cases = {}
    picks = {}
    for cls in ["healthy", "onychomycosis", "psoriasis"]:
        src_dir = HERE / "images" / "train" / cls
        cands = [p for p in sorted(src_dir.iterdir())
                 if p.is_file() and p.suffix.lower() in IMG_EXTS] if src_dir.is_dir() else []
        if not cands:
            print(f"[x] 工作区里没有 {cls} 图片, 请先运行 prepare_labeling.py", file=sys.stderr)
            return 2
        picks[cls] = cands[0]
        dst_dir = SCRATCH / "images" / "train" / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cands[0], dst_dir / f"{cls}_demo.jpg")
    print(f"  沙盒: {SCRATCH}")

    # ---------------- 2. 生成三类合成标注 ----------------
    print("\n[2/5] 生成合成 labelme JSON ...")
    # (a) healthy: 椭圆多边形, 独立用 PIL 光栅化做参考
    p = SCRATCH / "images" / "train" / "healthy" / "healthy_demo.jpg"
    with Image.open(p) as im:
        W, H = im.size
    cx, cy, rx, ry = W * 0.5, H * 0.5, W * 0.30, H * 0.24
    poly = ellipse_poly(cx, cy, rx, ry, 72)
    clean_meta = make_json(p, [shape("healthy", poly)])
    (p.with_suffix(".json")).write_text(json.dumps(clean_meta, ensure_ascii=False), encoding="utf-8")
    cases["healthy"] = {}

    # (b) onychomycosis: 矩形 + 中文别名标签
    p = SCRATCH / "images" / "train" / "onychomycosis" / "onychomycosis_demo.jpg"
    with Image.open(p) as im:
        W, H = im.size
    x0, y0 = int(W * 0.2), int(H * 0.25)
    x1, y1 = int(W * 0.8), int(H * 0.75)
    (p.with_suffix(".json")).write_text(json.dumps(
        make_json(p, [shape("甲癣", [(x0, y0), (x1, y1)], "rectangle")]), ensure_ascii=False),
        encoding="utf-8")
    cases["onychomycosis"] = {"rect": (x0, y0, x1, y1)}

    # (b2) onychomycosis 第二张: shape_type="mask" —— labelme 7 的 SAM 画笔/AI 辅助标注产物
    #      mask 是内嵌的 base64 PNG 补丁, 需要按 bbox 贴回, 是最容易实现错的一类
    src2 = [q for q in sorted((HERE / "images" / "train" / "onychomycosis").iterdir())
            if q.suffix.lower() in IMG_EXTS][1]
    p2 = SCRATCH / "images" / "train" / "onychomycosis" / "mask_demo.jpg"
    shutil.copy2(src2, p2)
    with Image.open(p2) as im:
        W2, H2 = im.size
    patch = np.zeros((30, 40), np.uint8)
    patch[5:25, 10:30] = 255                      # 30x40 补丁里嵌一个 20x20 方块
    buf = io.BytesIO()
    Image.fromarray(patch).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    bx, by = int(W2 * 0.15), int(H2 * 0.55)
    shp = shape("onychomycosis", [(bx, by), (bx + 39, by + 29)])
    shp["shape_type"] = "mask"
    shp["mask"] = b64
    (p2.with_suffix(".json")).write_text(json.dumps(
        make_json(p2, [shp]), ensure_ascii=False), encoding="utf-8")
    cases["mask"] = {"origin": (bx, by), "block": (5, 10, 25, 30),
                     "shape": (30, 40), "expected_px": 20 * 20}

    # (c) psoriasis: 类间覆盖(先画病灶矩形, 再用背景多边形盖掉右半边) + 圆 + 点 + 线段
    p = SCRATCH / "images" / "train" / "psoriasis" / "psoriasis_demo.jpg"
    with Image.open(p) as im:
        W, H = im.size
    x0, y0 = int(W * 0.15), int(H * 0.25)
    x1, y1 = int(W * 0.85), int(H * 0.60)
    xm = (x0 + x1) // 2
    ccx, ccy, r = W * 0.35, H * 0.78, min(W, H) * 0.10
    pcx, pcy = W * 0.80, H * 0.82
    ls = [(W * 0.06, H * 0.06), (W * 0.34, H * 0.09)]
    (p.with_suffix(".json")).write_text(json.dumps(make_json(p, [
        shape("psoriasis", [(x0, y0), (x1, y1)], "rectangle"),
        shape("psoriasis", [(ccx, ccy), (ccx + r, ccy)], "circle"),
        shape("psoriasis", [(pcx, pcy)], "point"),
        shape("背景", [(xm, y0), (x1, y0), (x1, y1), (xm, y1)]),
        shape("psoriasis", ls, "linestrip"),
    ]), ensure_ascii=False), encoding="utf-8")
    cases["psoriasis"] = {"rect": (x0, y0, x1, y1), "xm": xm,
                          "circle": (ccx, ccy, r), "point": (pcx, pcy), "line": ls}

    # 再放一张没有 json 的图, 检验"未标注"清单
    p2 = SCRATCH / "images" / "train" / "psoriasis" / "psoriasis_unlabeled.jpg"
    shutil.copy2(p, p2)

    # ---------------- 3. 跑转换 ----------------
    print("\n[3/5] 运行 labelme2mask.py ...")
    cmd = [sys.executable, str(TOOLS / "labelme2mask.py"), "--root", str(SCRATCH),
           "--splits", "train", "--vis"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("  " + "\n  ".join(r.stdout.strip().splitlines()[-14:]))
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
    check(r.returncode == 0, f"转换脚本退出码 = {r.returncode}")

    # ---------------- 4. 独立校验 ----------------
    print("\n[4/5] 独立校验掩膜 ...")
    mroot = SCRATCH / "masks" / "train"

    # (a) healthy: 椭圆多边形。转换器用 PIL 光栅化, 这里改用 OpenCV 独立光栅化做参照
    mp = mroot / "healthy" / "healthy_demo.png"
    m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
    check(m is not None and m.ndim == 2 and m.dtype == np.uint8,
          f"healthy: 掩膜为单通道 uint8 ({None if m is None else (m.ndim, m.dtype)})")
    if m is not None:
        with Image.open(SCRATCH / "images" / "train" / "healthy" / "healthy_demo.jpg") as im:
            check(m.shape == (im.size[1], im.size[0]), f"healthy: 掩膜尺寸与原图一致 {m.shape}")
        got = m == 1
        ref_cv = np.zeros_like(m)
        cv2.fillPoly(ref_cv, [np.round(poly).astype(np.int32)], 1, lineType=cv2.LINE_8)
        exp = ref_cv.astype(bool)
        iou = (got & exp).sum() / max((got | exp).sum(), 1)
        check(iou > 0.95, f"healthy: 与 OpenCV 独立光栅化多边形 IoU = {iou:.4f} > 0.95")
        # 面积也要与解析椭圆面积吻合(交叉验证两套光栅化都没跑偏)
        area_exp = math.pi * rx * ry
        check(abs(int(got.sum()) - area_exp) / area_exp < 0.05,
              f"healthy: 椭圆像元 {int(got.sum())} ≈ 解析面积 {area_exp:.0f} (误差<5%)")

    # (b) onychomycosis: 矩形面积精确(含端点, 与 labelme 官方 draw.rectangle 一致), 中文别名可解析
    mp = mroot / "onychomycosis" / "onychomycosis_demo.png"
    m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
    x0, y0, x1, y1 = cases["onychomycosis"]["rect"]
    if m is not None:
        got = int((m == 2).sum())
        exp = (x1 - x0 + 1) * (y1 - y0 + 1)      # 闭区间: 两端像素都算
        check(got == exp, f"onychomycosis: 矩形像元数 {got} == 期望(含端点) {exp}")
        check(set(np.unique(m).tolist()) == {0, 2},
              f"onychomycosis: 掩膜取值为 {np.unique(m).tolist()} (应为[0,2])")
        check(m[y0 + 2, x0 + 2] == 2 and m[y1 + 2, x1 + 2] == 0, "onychomycosis: 矩形内外像素类别正确")

    # (b2) mask 形状: 精确贴回 bbox 内
    mp = mroot / "onychomycosis" / "mask_demo.png"
    m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
    c = cases["mask"]
    if m is not None:
        bx, by = c["origin"]
        y1, x1, y2, x2 = c["block"]
        got = int((m == 2).sum())
        check(got == c["expected_px"],
              f"mask形状: 贴回的像素数 {got} == 期望 {c['expected_px']}")
        check(m[by + y1 + 1, bx + x1 + 1] == 2 and m[by + y1 - 2, bx + x1 - 2] == 0,
              "mask形状: 补丁内 20x20 方块位置正确、方块外为背景")
        check(set(np.unique(m).tolist()) == {0, 2}, f"mask形状: 掩膜取值 {np.unique(m).tolist()}")

    # (c) psoriasis: 覆盖顺序 + 圆 + 点 + 线段
    mp = mroot / "psoriasis" / "psoriasis_demo.png"
    m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
    c = cases["psoriasis"]
    if m is not None:
        x0, y0, x1, y1 = c["rect"]
        xm = c["xm"]
        ccx, ccy, r = c["circle"]
        pcx, pcy = c["point"]
        left_exp = (xm - x0) * (y1 - y0)
        got_left = int((m[y0:y1, x0:xm] == 3).sum())
        got_right = int((m[y0:y1, xm:x1] > 0).sum())
        check(got_left == left_exp, f"psoriasis: 矩形左半 3 类像元 {got_left} == 期望 {left_exp}")
        check(got_right == 0, f"psoriasis: 被后画背景多边形覆盖的右半前景像元 = {got_right} (应为 0, 后画覆盖先画)")
        check(m[int(ccy), int(ccx)] == 3, "psoriasis: 圆形中心类别 = 3")
        # 只在圆的包围盒内统计, 避开线段与点
        ry0, ry1 = max(int(ccy - r) - 2, 0), min(int(ccy + r) + 3, m.shape[0])
        rx0, rx1 = max(int(ccx - r) - 2, 0), min(int(ccx + r) + 3, m.shape[1])
        circle_px = int((m[ry0:ry1, rx0:rx1] == 3).sum())
        circle_exp = math.pi * r * r
        check(abs(circle_px - circle_exp) / circle_exp < 0.10,
              f"psoriasis: 圆形像元 {circle_px} ≈ 解析面积 {circle_exp:.0f} (误差<10%)")
        check(m[int(pcy), int(pcx)] == 3, "psoriasis: 点标注类别 = 3")
        check(set(np.unique(m).tolist()) <= {0, 3}, f"psoriasis: 掩膜取值 {np.unique(m).tolist()} ⊆ {{0,3}}")
        # 线段: 用「到线段的几何距离」独立验证, 而不是拿另一种光栅化去比 IoU
        a, b = c["line"][0], c["line"][1]
        mg = 16
        sy0, sy1 = max(int(min(a[1], b[1])) - mg, 0), min(int(max(a[1], b[1])) + mg, m.shape[0])
        sx0, sx1 = max(int(min(a[0], b[0])) - mg, 0), min(int(max(a[0], b[0])) + mg, m.shape[1])
        sub = m[sy0:sy1, sx0:sx1] == 3
        ys, xs = np.nonzero(sub)
        if len(xs) == 0:
            check(False, "psoriasis: linestrip 没有画出任何像素")
        else:
            pts = np.stack([xs + sx0, ys + sy0], axis=1).astype(float)
            d = seg_dist(pts, a, b)
            check(d.max() <= 6.0, f"psoriasis: linestrip 所有像元距线段 ≤6px (实测最远 {d.max():.1f}px)")
            # 覆盖度: 沿线段每 0.5px 取样, 每个取样点 3px 邻域内都应有线段像元
            n_s = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) * 2), 2)
            ts = np.linspace(0, 1, n_s)[:, None]
            samples = np.asarray(a)[None, :] + ts * (np.asarray(b) - np.asarray(a))[None, :]
            covered = 0
            for s in samples:
                x, y = int(round(s[0])), int(round(s[1]))
                win = m[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4]
                covered += int((win == 3).any())
            frac = covered / len(samples)
            check(frac > 0.95, f"psoriasis: linestrip 沿线覆盖度 {frac:.1%} > 95%")

    # (d) 未标注清单
    miss_f = SCRATCH / "qc" / "missing_annotations.txt"
    miss = miss_f.read_text(encoding="utf-8") if miss_f.is_file() else ""
    check("psoriasis_unlabeled.jpg" in miss, "未标注图片被写入 qc/missing_annotations.txt")
    check((SCRATCH / "qc" / "convert_report.txt").is_file(), "转换报告已生成")

    # (e) 类别不符与未知标签的报错路径
    bad = SCRATCH / "images" / "train" / "healthy" / "healthy_demo.json"
    meta = json.loads(bad.read_text(encoding="utf-8"))
    meta["shapes"].append(shape("不存在的类", [(10, 10), (40, 10), (40, 40)]))
    bad.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    r2 = subprocess.run(cmd[:-1], capture_output=True, text=True, input=None)
    check("无法识别的标签" in (r2.stdout + r2.stderr), "未知标签在默认 --unknown error 下会明确报错")
    r3 = subprocess.run(cmd[:-1] + ["--unknown", "ignore"], capture_output=True, text=True)
    check(r3.returncode == 0, "未知标签在 --unknown ignore 下可跳过并继续")
    meta["shapes"][-1]["label"] = "psoriasis"      # 类别与文件夹(healthy)不符
    bad.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    subprocess.run(cmd[:-1], capture_output=True, text=True)
    mm_f = SCRATCH / "qc" / "class_mismatch.txt"
    mm = mm_f.read_text(encoding="utf-8") if mm_f.is_file() else ""
    check("类别与文件夹不符" in mm, "多边形标签与文件夹类别不符会被单独记录")

    # ---------------- 5. 生成肉眼核对图 ----------------
    print("\n[5/5] 生成三联叠加图 ...")
    # 上一步故意写坏了 healthy 的 json, 这里恢复成干净版本再转一次
    bad.write_text(json.dumps(clean_meta, ensure_ascii=False), encoding="utf-8")
    subprocess.run(cmd, capture_output=True, text=True)
    tiles = []
    for cls in ["healthy", "onychomycosis", "psoriasis"]:
        ip = SCRATCH / "images" / "train" / cls / f"{cls}_demo.jpg"
        mp = mroot / cls / f"{cls}_demo.png"
        img = cv2.imread(str(ip))
        if img is None or not mp.is_file():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        color = np.zeros((*m.shape, 3), np.uint8)
        color[m == 1] = (113, 204, 46)     # BGR: healthy 绿
        color[m == 2] = (15, 196, 241)     # onychomycosis 黄
        color[m == 3] = (60, 76, 231)      # psoriasis 红
        ov = cv2.addWeighted(img, 0.55, color, 0.45, 0)
        s = 300
        row = [cv2.resize(x, (s, s), interpolation=cv2.INTER_AREA) for x in (img, color, ov)]
        row = [np.hstack([r, np.full((s, 4, 3), 255, np.uint8)]) for r in row]
        strip = np.hstack(row)
        cv2.putText(strip, cls, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(strip)
    if tiles:
        wmin = min(t.shape[1] for t in tiles)
        canvas = np.vstack([t[:, :wmin] for t in tiles])
        out = HERE / "qc" / "selftest_overlay.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)
        print(f"  已写出: {out}")

    # ---------------- 汇总 ----------------
    n_fail = sum(1 for ok, _ in RESULTS if not ok)
    print("\n" + "=" * 72)
    print(f"自测结果: {len(RESULTS) - n_fail}/{len(RESULTS)} 通过" +
          ("" if n_fail == 0 else f", {n_fail} 项失败"))
    for ok, msg in RESULTS:
        if not ok:
            print(f"  失败: {msg}")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

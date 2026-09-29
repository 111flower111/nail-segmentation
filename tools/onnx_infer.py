#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onnx_infer.py —— ONNX 模型推理参考实现（手机端照抄这份逻辑即可）
====================================================================

它用的预处理**与训练/导出时严格一致**，是 App 端必须遵守的约定：
    读图(OpenCV 给的是 BGR)
      -> cv2.resize 到 512x512   (直接拉伸, 不保持长宽比, 与训练一致)
      -> BGR 转 RGB              (这一步最容易漏!)
      -> HWC -> CHW, float32
      -> 喂进 ONNX

同时可以顺带在验证集上算 IoU，用来确认"走 App 这条路"精度没有掉。

用法:
    # 跑若干张图, 输出三联图 原图|彩色预测|叠加
    python3 onnx_infer.py --model deploy/nail_3class_512.onnx \
        --out-dir vis_onnx "images/val/onychomycosis/*.jpg"

    # 顺便按 GT 掩膜算前景 IoU（掩膜路径由 images/ 自动替换为 masks3/）
    python3 onnx_infer.py --model deploy/nail_3class_512.onnx \
        --out-dir vis_onnx --masks-root masks3 \
        "images/val/*/*.jpg"
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
NAMES = ['background', 'healthy', 'diseased']
PALETTE_RGB = np.array([[0, 0, 0], [46, 204, 113], [231, 76, 60]], np.uint8)


def build_session(model_path: str, threads: int = 0) -> ort.InferenceSession:
    so = ort.SessionOptions()
    if threads > 0:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=so,
                                providers=['CPUExecutionProvider'])


def infer(sess, in_name: str, bgr: np.ndarray, size: int = 512,
          want_logits: bool = False):
    """输入 BGR 图(任意尺寸) -> (类别图 uint8[原图尺寸], logits 或 None)"""
    h0, w0 = bgr.shape[:2]
    x = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    x = x[:, :, ::-1].astype(np.float32)                 # ★ BGR -> RGB
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    outs = sess.run(None, {in_name: x})
    logits, pred = outs[0], outs[1] if len(outs) > 1 else (outs[0], None)
    if pred is None:
        pred = logits.argmax(1).astype(np.uint8)
    pred = pred[0]
    if (h0, w0) != (size, size):
        pred = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return pred, (logits[0] if want_logits else None)


def mask_path_for(img_path: str, masks_root: str) -> str | None:
    """由图片路径推出 GT 掩膜路径: 把路径里的 'images' 段换成 masks_root, 后缀换成 .png。

    不用字符串 replace, 因为相对路径(images/val/..)和绝对路径(/home/../images/val/..)
    的前缀形态不同, replace 很容易匹配不到。
    """
    parts = Path(img_path).parts
    if 'images' not in parts:
        return None
    i = parts.index('images')
    rel = parts[i + 1:]
    if not rel:
        return None
    return str(Path(masks_root).joinpath(*rel).with_suffix('.png'))


def confusion_counts(pred: np.ndarray, gt: np.ndarray, n_cls: int = 3):
    """返回逐类的 (tp, fp, fn) 计数。累加后再算 IoU 才与 MMSeg 口径一致——
    逐图求平均是错的(小图和大图权重不同)。"""
    tp = np.zeros(n_cls, np.int64)
    fp = np.zeros(n_cls, np.int64)
    fn = np.zeros(n_cls, np.int64)
    for c in range(n_cls):
        p, g = (pred == c), (gt == c)
        tp[c] = int((p & g).sum())
        fp[c] = int((p & ~g).sum())
        fn[c] = int((~p & g).sum())
    return tp, fp, fn


def metrics_from_counts(tp, fp, fn):
    """由累加的 tp/fp/fn 计算逐类 IoU / 召回 / Dice (无定义处返回 nan)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(tp + fp + fn > 0, tp / np.maximum(tp + fp + fn, 1), np.nan)
        rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan)
        dice = np.where(2 * tp + fp + fn > 0,
                        2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    return iou * 100, rec * 100, dice * 100


def main() -> int:
    ap = argparse.ArgumentParser(description='ONNX 推理 + 可视化 + 可选 IoU',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('paths', nargs='+', help='图片路径或 glob')
    ap.add_argument('--out-dir', default='vis_onnx')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--masks-root', default='', help='GT 掩膜根目录(如 masks3), 给了就算 IoU')
    ap.add_argument('--limit', type=int, default=0, help='最多处理多少张(0=全部)')
    ap.add_argument('--threads', type=int, default=0, help='onnxruntime 线程数(0=默认)')
    args = ap.parse_args()

    files: list[str] = []
    for pat in args.paths:
        if os.path.isdir(pat):
            for ext in IMG_EXTS:
                files += glob.glob(os.path.join(pat, '**', '*' + ext), recursive=True)
        else:
            files += glob.glob(pat)
    files = sorted(set(f for f in files if f.lower().endswith(IMG_EXTS)))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print('[x] 没有匹配到图片')
        return 2

    sess = build_session(args.model, args.threads)
    in_name = sess.get_inputs()[0].name
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f'模型   : {args.model}')
    print(f'输入   : {in_name} {sess.get_inputs()[0].shape}')
    print(f'图片   : {len(files)} 张  ->  {out_dir}/')

    TP = np.zeros(3, np.int64); FP = np.zeros(3, np.int64); FN = np.zeros(3, np.int64)
    n_eval = 0
    times = []
    for p in files:
        img = cv2.imread(p)
        if img is None:
            print(f'  [跳过] 读不了 {p}'); continue
        t0 = time.time()
        pred, _ = infer(sess, in_name, img, args.size)
        dt = time.time() - t0
        times.append(dt)

        color = PALETTE_RGB[pred][:, :, ::-1]             # RGB -> BGR 供 OpenCV 写图
        ov = cv2.addWeighted(img, 0.55, color, 0.45, 0)
        out = out_dir / (Path(p).stem + '_onnx.jpg')
        cv2.imwrite(str(out), np.hstack([img, color, ov]))

        stat = {NAMES[i]: f'{(pred == i).mean()*100:5.1f}%' for i in range(3) if (pred == i).any()}
        line = f'  {Path(p).name[:44]:46s} {dt*1000:6.1f}ms  {stat}'

        # 有 GT 掩膜就算 IoU
        if args.masks_root:
            mp = mask_path_for(p, args.masks_root)
            if mp and os.path.isfile(mp):
                gt = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
                if gt is not None and gt.shape == pred.shape:
                    t, f, n = confusion_counts(pred, gt, 3)
                    TP += t; FP += f; FN += n; n_eval += 1
                    iou_c, _, _ = metrics_from_counts(t, f, n)
                    fg = np.nanmean([iou_c[1], iou_c[2]])
                    line += f'  | 本图前景IoU {fg:5.1f}%'
        print(line)

    print(f'\n平均耗时 {np.mean(times)*1000:.1f} ms/张  ({1/np.mean(times):.1f} FPS)  '
          f'[本机 CPU, 线程={args.threads or "默认"}]')
    if n_eval:
        iou_c, rec_c, dice_c = metrics_from_counts(TP, FP, FN)
        print(f'\n--- 走 App 预处理路径的验证集指标 ({n_eval} 张, 累加混淆矩阵, 与 MMSeg 同口径) ---')
        print(f'{"类别":<12}{"IoU":>8}{"召回":>8}{"Dice":>8}')
        for i, nm in enumerate(NAMES):
            print(f'{nm:<12}{iou_c[i]:>8.2f}{rec_c[i]:>8.2f}{dice_c[i]:>8.2f}')
        fg = np.nanmean([iou_c[1], iou_c[2]])
        print(f'{"前景 mIoU":<12}{fg:>8.2f}   (healthy 与 diseased 的平均)')
        print(f'  (参考: MMSeg 训练时验证集 best_mIoU_iter_800 的 mIoU=88.71, 前景=(82.36+87.08)/2=84.72)')
    print(f'\n对比图已写到 {out_dir}/  (左=原图 中=预测类别图 右=叠加)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

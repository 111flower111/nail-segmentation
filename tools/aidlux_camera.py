#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aidlux_camera.py —— AidLux 手机端：摄像头实时指甲分割
========================================================

前置条件:
    1. 把 nail_3class_512.onnx 拷到手机(AidLux 里), 路径见下面的 MODEL
    2. AidLux 终端里装好:  pip install onnxruntime==1.19.2

预处理必须与训练/导出严格一致(与 onnx_infer.py 同一套):
    OpenCV 读帧(或从 AidLux 相机拿) -> 是 BGR
      -> cv2.resize 到 512x512 (直接拉伸, 不保持长宽比)
      -> BGR -> RGB          ★ 最容易漏的一步
      -> HWC -> CHW, float32
      -> onnxruntime 推理
      -> 预测图 resize 回屏幕尺寸(最近邻)

用法:
    python3 aidlux_camera.py                 # 摄像头实时(默认)
    python3 aidlux_camera.py --save-every 30 # 每 30 帧存一张结果图, 便于无显示环境下验证
    python3 aidlux_camera.py --headless      # 不尝试弹窗(AidLux 没有 X 时用)
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

MODEL = '/home/aidlux/nail_3class_512.onnx'   # ← 按实际路径改
SIZE = 512
NAMES = ['background', 'healthy', 'diseased']
PALETTE_RGB = np.array([[0, 0, 0], [46, 204, 113], [231, 76, 60]], np.uint8)
# 屏幕显示用的 BGR 配色
PALETTE_BGR = PALETTE_RGB[:, ::-1].copy()


def open_camera(index: int = 0):
    """优先用 OpenCV 开摄像头; 不行再退回 AidLux 自带的 cvs 模块。"""
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print('[ok] 用 cv2.VideoCapture 打开摄像头')
        return ('cv2', cap)

    try:
        import cvs  # AidLux 自带
        print('[ok] 用 AidLux cvs 模块打开摄像头')
        return ('cvs', cvs.VideoCapture(index))
    except Exception as e:                       # noqa: BLE001
        raise RuntimeError(
            f'摄像头打不开。cv2.VideoCapture 失败, cvs 也不可用: {e}\n'
            '检查: 1) AidLux 是否有相机权限  2) 摄像头是否被其他 App 占用') from e


def read_frame(kind, cap):
    if kind == 'cv2':
        ok, frame = cap.read()
        return frame if ok else None
    return cap.read()                            # cvs.VideoCapture.read() 直接返回帧


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--save-dir', default='captures')
    ap.add_argument('--save-every', type=int, default=0, help='每 N 帧存一张(0=不存)')
    ap.add_argument('--headless', action='store_true', help='不尝试弹窗')
    ap.add_argument('--threads', type=int, default=0, help='onnxruntime 线程数(0=默认)')
    ap.add_argument('--warmup', type=int, default=3)
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f'[x] 找不到模型: {args.model}'); return 2

    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.model, sess_options=so,
                                providers=['CPUExecutionProvider'])
    IN = sess.get_inputs()[0].name
    print(f'[ok] 模型已加载: {args.model}')
    print(f'     输入 {sess.get_inputs()[0].shape}, 输出 {[o.name for o in sess.get_outputs()]}')

    kind, cap = open_camera(args.camera)
    if args.save_every:
        os.makedirs(args.save_dir, exist_ok=True)

    n, fps_t, fps = 0, time.time(), 0.0
    while True:
        frame = read_frame(kind, cap)
        if frame is None:
            print('[!] 读帧失败, 退出'); break

        x = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        x = x[:, :, ::-1].astype(np.float32)              # ★ BGR -> RGB
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])

        t0 = time.time()
        outs = sess.run(None, {IN: x})
        infer_ms = (time.time() - t0) * 1000
        pred = outs[1][0] if len(outs) > 1 else outs[0].argmax(1)[0].astype(np.uint8)

        h, w = frame.shape[:2]
        color = PALETTE_BGR[cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)]
        vis = cv2.addWeighted(frame, 0.55, color, 0.45, 0)

        # 结论: 病甲像素占比
        ill = float((pred == 2).mean())
        ok_health = float((pred == 1).mean())
        if ill > 0.02:
            txt, col = f'DISEASED {ill*100:.1f}%', (0, 0, 255)
        elif ok_health > 0.05:
            txt, col = f'HEALTHY {ok_health*100:.1f}%', (0, 200, 0)
        else:
            txt, col = 'NO NAIL DETECTED', (0, 165, 255)
        cv2.putText(vis, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, col, 2)
        cv2.putText(vis, f'{infer_ms:.0f}ms  {fps:.1f}FPS', (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        n += 1
        if time.time() - fps_t >= 1.0:
            fps = n / (time.time() - fps_t); n = 0; fps_t = time.time()
            print(f'  {infer_ms:6.1f} ms/帧  {fps:5.1f} FPS  {txt}')

        if args.save_every and n % args.save_every == 0:
            p = os.path.join(args.save_dir, f'cap_{int(time.time())}.jpg')
            cv2.imwrite(p, np.hstack([frame, color, vis]))
            print(f'  [存图] {p}')

        if not args.headless:
            try:
                cv2.imshow('nail', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:                    # noqa: BLE001
                print(f'[!] 无法弹窗({e}), 转为 headless 模式(只打印/存图)')
                args.headless = True

    try:
        cap.release()
    except Exception:                                 # noqa: BLE001
        pass
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

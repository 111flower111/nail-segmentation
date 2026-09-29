#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_onnx.py —— 把训练好的 MMSegmentation 模型导出成自包含的 ONNX
======================================================================

为什么不用 MMDeploy:
    目标后端只有 ONNX Runtime(手机端), MMDeploy 要额外编译 C 扩展、版本敏感,
    而它导出的 ONNX 里并不含预处理, 还得在 App 里手工复刻归一化, 反而更容易错。
    本脚本直接把「预处理 + 网络 + 上采样 + argmax」全部内联进计算图,
    App 端只需要: 读图 -> resize 到 512x512 -> 转 RGB -> float32 -> 喂进去。

接口约定(务必与 App 端一致):
    输入  input : float32 [1, 3, H, W]   RGB 顺序, 取值范围 [0, 255]
    输出  logits: float32 [1, C, H, W]   未归一化的原始分数(想调阈值就用它)
    输出  pred  : uint8   [1, H, W]      argmax 后的类别图(0=背景 1=健康 2=病甲)

    为什么输入是 RGB 而不是 BGR:
        MMSeg 的流程是 读图(BGR) -> bgr_to_rgb(变 RGB) -> 减均值除标准差。
        两步合起来等价于「RGB 图直接减同一组均值」。本脚本按等价形式内联,
        所以这里按 RGB 传即可, 不要传 BGR。

用法:
    # 导出 + 与 PyTorch/MMSeg 官方推理做数值一致性验证
    python3 export_onnx.py \
        --config configs/segformer-b0_nail_3class_512.py \
        --checkpoint work_dirs/nail_3class/best_mIoU_iter_800.pth \
        --out deploy/nail_3class_512.onnx

    # 试算力(纯 onnxruntime 前向耗时)
    python3 export_onnx.py ... --bench 50
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config

# 与 MMSeg 官方 SegFormer 配置保持一致的 ImageNet 均值/标准差(**RGB 顺序**)
IMG_MEAN = [123.675, 116.28, 103.53]
IMG_STD = [58.395, 57.12, 57.375]


class SegOnnxWrapper(torch.nn.Module):
    """把 MMSeg 的 EncoderDecoder 包成「图像 -> 类别图」的自包含模型。

    等价于 MMSeg 推理链路的:
        LoadImageFromFile -> Resize -> SegDataPreProcessor(bgr_to_rgb + normalize)
        -> backbone -> decode_head.forward -> predict_by_feat(bilinear resize)
        -> argmax
    其中 bgr_to_rgb 与 normalize 合并为「对 RGB 直接做 (x-mean)/std」。
    """

    def __init__(self, model, return_argmax: bool = True):
        super().__init__()
        self.model = model
        self.return_argmax = return_argmax
        self.register_buffer('mean',
                             torch.tensor(IMG_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std',
                             torch.tensor(IMG_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor):
        x = (x - self.mean) / self.std
        # 绕开 data_preprocessor(否则会二次归一化); _forward = backbone + decode_head
        logits = self.model._forward(x)
        # 与 decode_head.predict_by_feat 一致: bilinear + align_corners=False
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear',
                               align_corners=False)
        if self.return_argmax:
            return logits, logits.argmax(dim=1).to(torch.uint8)
        return logits


def load_model(config: str, checkpoint: str, device: str = 'cpu'):
    from mmseg.apis import init_model
    from mmseg.utils import register_all_modules
    register_all_modules()
    model = init_model(config, checkpoint, device=device)
    model.eval()
    return model


def preprocess_rgb(img_bgr: np.ndarray, size) -> np.ndarray:
    """按 MMSeg 的 Resize(keep_ratio=False) 等价方式处理, 返回 NCHW float32 RGB。"""
    import mmcv
    resized = mmcv.imresize(img_bgr, size)          # 与 pipeline 里的 Resize 同一实现
    rgb = resized[:, :, ::-1].astype(np.float32)    # BGR -> RGB
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def main() -> int:
    ap = argparse.ArgumentParser(description='MMSeg -> 自包含 ONNX',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True, help='输出 .onnx 路径')
    ap.add_argument('--shape', default='512,512', help='固定输入尺寸 H,W')
    ap.add_argument('--dynamic', action='store_true',
                    help='H/W 设为动态。⚠ 实测不可靠: MixVisionTransformer 的 '
                         'PatchEmbed 用 x.shape 推导 H/W, 追踪时会固化成常量, '
                         '动态轴形同虚设。需要多分辨率请分别导出多个固定尺寸模型。')
    ap.add_argument('--opset', type=int, default=13,
                    help='SegFormer 的 MixVisionTransformer 用了 aten::unflatten，'
                         '该算子 opset>=13 才支持；若仍报不支持可试 17')
    ap.add_argument('--no-argmax', action='store_true', help='只输出 logits')
    ap.add_argument('--verify-images', default='', help='用于一致性验证的图片目录(逗号分隔)')
    ap.add_argument('--verify-n', type=int, default=5)
    ap.add_argument('--bench', type=int, default=0, help='基准测试前向次数(纯 onnxruntime)')
    args = ap.parse_args()

    H, W = [int(x) for x in args.shape.split(',')]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    classes = list(cfg.metainfo['classes'])
    num_classes = cfg.model['decode_head']['num_classes']
    print('=' * 74)
    print('导出 MMSeg 模型 -> ONNX')
    print('=' * 74)
    print(f'config     : {args.config}')
    print(f'checkpoint : {args.checkpoint}')
    print(f'类别       : {classes} (num_classes={num_classes})')
    print(f'输入尺寸   : {"动态" if args.dynamic else f"固定 {H}x{W}"}')

    model = load_model(args.config, args.checkpoint)
    wrapper = SegOnnxWrapper(model, return_argmax=not args.no_argmax).eval()

    dummy = torch.randn(1, 3, H, W)
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {'input': {2: 'height', 3: 'width'},
                        'logits': {2: 'height', 3: 'width'}}
        if not args.no_argmax:
            dynamic_axes['pred'] = {1: 'height', 2: 'width'}
    out_names = ['logits'] if args.no_argmax else ['logits', 'pred']

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(out_path),
            input_names=['input'], output_names=out_names,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            export_params=True,
        )
    size_mb = out_path.stat().st_size / 1e6
    print(f'\n[ok] 已写出 {out_path}  ({size_mb:.1f} MB)')

    # ---------------------------------------------------------------- 验证
    import onnx
    onnx.checker.check_model(onnx.load(str(out_path)))
    print('[ok] onnx.checker 通过')

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    print(f'[ok] onnxruntime {ort.__version__} 加载成功, 输入名={in_name}')
    print(f'     输入 {sess.get_inputs()[0].type} {sess.get_inputs()[0].shape}')
    for o in sess.get_outputs():
        print(f'     输出 {o.name:7s} {o.type} {o.shape}')

    if args.verify_images:
        from mmseg.apis import inference_model
        import glob, os
        imgs = []
        for d in args.verify_images.split(','):
            d = d.strip()
            if os.path.isdir(d):
                imgs += sorted(glob.glob(os.path.join(d, '**', '*.jpg'), recursive=True))
        imgs = imgs[:args.verify_n]
        print(f'\n--- 数值一致性验证 ({len(imgs)} 张) ---')
        print('    (ONNX≡Torch 在 512x512 上比; ONNX≡MMSeg官方 在**原图尺寸**上比,')
        print('     因为 MMSeg 是把 logits 上采样到原图后再 argmax 的)')
        worst1 = worst2 = 0.0
        ag1, ag2 = [], []
        for p in imgs:
            img = cv2.imread(p)
            h0, w0 = img.shape[:2]
            x = preprocess_rgb(img, (W, H))

            # -- PyTorch wrapper --
            with torch.no_grad():
                out_pt = wrapper(torch.from_numpy(x))
            lg_pt = out_pt[0].numpy()
            pd_pt = out_pt[1].numpy()[0] if len(out_pt) > 1 else lg_pt.argmax(1)[0].astype(np.uint8)

            # -- ONNX Runtime --
            outs = sess.run(None, {in_name: x})
            lg_ort = outs[0]
            pd_ort = outs[1][0] if len(outs) > 1 else lg_ort.argmax(1)[0].astype(np.uint8)

            d1 = float(np.abs(lg_pt - lg_ort).max())
            a1 = float((pd_pt == pd_ort).mean() * 100)

            # -- MMSeg 官方链路(原图尺寸) --
            res = inference_model(model, p)
            ref_pred = res.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
            ref_logits = res.seg_logits.data.cpu().numpy()
            lg_up = F.interpolate(torch.from_numpy(lg_ort), size=(h0, w0),
                                  mode='bilinear', align_corners=False).numpy()[0]
            d2 = float(np.abs(lg_up - ref_logits).max())
            a2 = float((lg_up.argmax(0).astype(np.uint8) == ref_pred).mean() * 100)

            worst1 = max(worst1, d1); worst2 = max(worst2, d2)
            ag1.append(a1); ag2.append(a2)
            print(f'  {os.path.basename(p)[:40]:42s} '
                  f'|512 logits差|={d1:.1e} Torch {a1:6.2f}% | '
                  f'|原图 logits差|={d2:.1e} MMSeg官方 {a2:6.2f}%')
        print(f'\n  512x512 : 最大 logits 偏差 {worst1:.2e} (应<1e-4), 最低一致率 {min(ag1):.2f}%')
        print(f'  原图尺寸: 最大 logits 偏差 {worst2:.2e} (应<1e-4), 最低一致率 {min(ag2):.2f}%')

    if args.bench:
        img = np.random.rand(1, 3, H, W).astype(np.float32) * 255
        # 基准只测第一个输出(logits), 避免多输出拷贝干扰
        for _ in range(3):
            sess.run(None, {in_name: img})
        t0 = time.time()
        for _ in range(args.bench):
            sess.run(None, {in_name: img})
        dt = (time.time() - t0) / args.bench
        print(f'\n--- 算力基准 (本机 CPU, 仅供相对参考) ---')
        print(f'   {dt*1000:.1f} ms/张  ({1/dt:.1f} FPS)  输入 {H}x{W}')

    print('\n' + '=' * 74)
    print('交付给 App 端的接口约定:')
    print(f'  输入  input  float32 [1,3,{H},{W}]  RGB, [0,255]   (不要传 BGR!)')
    print(f'  输出  logits float32 [1,{num_classes},{H},{W}]     原始分数')
    if not args.no_argmax:
        print(f'  输出  pred   uint8   [1,{H},{W}]            类别图: ' +
              ', '.join(f'{i}={c}' for i, c in enumerate(classes)))
    print('=' * 74)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

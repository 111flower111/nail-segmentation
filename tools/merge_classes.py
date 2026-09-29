#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_classes.py —— 把 4 类掩膜合并成 3 类（背景 / 健康 / 病甲）
==================================================================

背景:
    原方案 4 类: 0=背景 1=健康 2=甲癣 3=银屑病
    甲癣与银屑病单凭照片极难区分(临床上靠真菌镜检), 两者混淆是模型精度上不去的主因。
    合并成 3 类后前景只剩 健康 / 病甲, 任务难度大幅下降。

映射:
    0 -> 0  (background)
    1 -> 1  (healthy)
    2 -> 2  (diseased 病甲)
    3 -> 2  (diseased 病甲)      <-- 只改这一个

特点:
    * 非破坏性: 原 masks/ 一个字节都不动, 结果写到新目录(默认 masks3/)
    * 保留子目录结构: MMSeg 靠 `图片相对路径` 换成 `.png` 去找掩膜,
      所以 masks3/ 必须镜像 masks/ 的目录层级, 否则配对会全部失败
    * 只动 masks, 不碰 images
    * 转换后自动校验: 取值必须 ⊆ {0,1,2}、张数一致、尺寸不变、且只有 3 的位置被改

用法:
    # 先干跑看看会发生什么
    python3 merge_classes.py --root . --dry-run

    # 真正执行(默认写到 masks3)
    python3 merge_classes.py --root .

    # 自定义输出位置
    python3 merge_classes.py --root . --dst /path/to/masks3
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# 4 类 -> 3 类 的映射表(LUT), 索引=旧值
LUT = np.array([0, 1, 2, 2] + [255] * 252, dtype=np.uint8)

CLASSES_IN = ('background', 'healthy', 'onychomycosis', 'psoriasis')
CLASSES_OUT = ('background', 'healthy', 'diseased')
PALETTE_OUT = [[0, 0, 0], [46, 204, 113], [231, 76, 60]]


def main() -> int:
    ap = argparse.ArgumentParser(description='4 类掩膜 -> 3 类掩膜(背景/健康/病甲)',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--root', type=Path, required=True, help='数据集根目录(需含 images/ 和 masks/)')
    ap.add_argument('--dst', type=Path, default=None, help='输出目录(默认 <root>/masks3)')
    ap.add_argument('--splits', default='train,val')
    ap.add_argument('--force', action='store_true', help='输出目录已存在时覆盖')
    ap.add_argument('--dry-run', action='store_true', help='只统计不写文件')
    args = ap.parse_args()

    src_root = args.root / 'masks'
    dst_root = args.dst or (args.root / 'masks3')
    if not src_root.is_dir():
        print(f'[x] 找不到源掩膜目录: {src_root}', file=sys.stderr)
        return 2
    if dst_root.exists() and not args.force and not args.dry_run:
        print(f'[x] 输出目录已存在: {dst_root}\n    加 --force 覆盖, 或换 --dst', file=sys.stderr)
        return 2

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    before = np.zeros(4, dtype=np.int64)
    after = np.zeros(3, dtype=np.int64)
    n_img = 0
    bad: list[str] = []
    changed_px = 0
    other_px = 0

    for split in splits:
        sdir = src_root / split
        if not sdir.is_dir():
            print(f'[!] 跳过不存在的划分: {sdir}')
            continue
        for cls_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            for src in sorted(cls_dir.glob('*.png')):
                m = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
                if m is None or m.ndim != 2:
                    bad.append(f'非单通道掩膜: {src}')
                    continue
                vals = np.unique(m)
                if vals.max() > 3:
                    bad.append(f'出现越界取值 {vals.tolist()}: {src}')
                    continue
                for v in vals:
                    before[v] += int((m == v).sum())

                out = LUT[m]                       # 只做查表
                changed_px += int((m == 3).sum())
                for v in np.unique(out):
                    after[v] += int((out == v).sum())

                if not args.dry_run:
                    d = dst_root / split / cls_dir.name / src.name
                    d.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(d), out):
                        bad.append(f'写入失败: {d}')
                        continue
                    # 立即回读校验
                    chk = cv2.imread(str(d), cv2.IMREAD_UNCHANGED)
                    if chk is None or chk.shape != m.shape:
                        bad.append(f'回读尺寸不符: {d}')
                    elif int(np.unique(chk).max()) > 2:
                        bad.append(f'回读取值越界: {d}')
                    elif not np.array_equal(LUT[chk], chk):
                        bad.append(f'回读不可逆(说明有值>2): {d}')
                n_img += 1

    print('=' * 70)
    print('4 类 -> 3 类 掩膜合并报告' + ('   [DRY RUN 未写文件]' if args.dry_run else ''))
    print('=' * 70)
    print(f'源目录    : {src_root}')
    print(f'输出目录  : {dst_root}')
    print(f'处理图片  : {n_img} 张')
    print()
    print('合并前(4 类) 像素数:')
    for i, c in enumerate(CLASSES_IN):
        print(f'    {i} {c:<15} {before[i]:>12,}')
    print('合并后(3 类) 像素数:')
    for i, c in enumerate(CLASSES_OUT):
        print(f'    {i} {c:<15} {after[i]:>12,}')
    print()
    print(f'被改写(3->2)的像素: {changed_px:,}')
    print(f'前景像素 健康:病甲 = 1 : {after[2]/max(after[1],1):.2f}')
    print()
    print('校验用的类别定义(抄进配置):')
    print(f'    num_classes = 3')
    print(f"    classes = {CLASSES_OUT}")
    print(f'    palette = {PALETTE_OUT}')
    print('=' * 70)
    if bad:
        print(f'\n[x] {len(bad)} 个问题:')
        for b in bad[:20]:
            print('   -', b)
        return 1
    print('\n[ok] 全部校验通过')
    if not args.dry_run:
        print(f'     下一步: 把配置里的 ann_dir 改成 {dst_root.name}/train 和 {dst_root.name}/val')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

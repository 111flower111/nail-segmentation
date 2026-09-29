# 指甲 3 类语义分割 —— SegFormer-B0 (MiT-B0) @ 512x512
# 类别: 0=background 1=healthy 2=diseased(病甲，甲癣+银屑病合并)
#
# 为什么合并: 甲癣与银屑病单凭照片极难区分(临床靠真菌镜检)，两者混淆是 4 类方案
#             精度上不去的主因(甲癣召回仅 34%、银屑病多报 5 倍面积)。合并后前景只剩
#             健康/病甲，任务难度大幅下降。
# 掩膜来源: masks3/ 由 tools/merge_classes.py 从 masks/ 生成(只把 3 改成 2，其余不动)。
#             目录结构必须镜像 images/，否则 MMSeg 配对会失败。
# 相对 v1 的改动（用于提升精度，可逐项 A/B 对照）：
#   ① 损失: 单 CE -> CE + Dice 组合（Dice 对前景小目标更友好）
#   ② 增强: 加 RandomRotate + RandomCutOut，并把颜色抖动调强（你的图来自不同手机/光照）
#   ③ 迭代: 3000 -> 1000（v1 实测 600 轮就到峰值，多跑是浪费）
#   ④ 验证更密: val_interval 100 -> 50（便于观察在哪一轮最好）
#   ⑤ 新增 TTA 配置，测试时加 --tta 可小幅涨点
# 原 v1 配置保留不动，方便对比实验。
# 环境: MMSegmentation 1.2.2 / mmengine 0.10.4 / mmcv 2.1.0 / torch 2.1.2+cu121
# 数据: images|masks/{train,val}/<类别>/  (165 张)
#
# 模型结构已与官方 configs/_base_/models/segformer_mit-b0.py 逐项核对一致；
# 两处有意差异：norm_cfg 用 BN（单卡）、lr 按 batch 线性缩放（6e-5 * 8/16 = 3e-5）。
default_scope = 'mmseg'

# mmseg 1.2.2 的 mmseg/__init__.py 只做版本检查、不导入任何子模块，
# 且它自带的 register_all_modules() 也不含 mmseg.visualization。
# 所以这里必须把要用到的子模块全部显式导入，否则会依次报
#   "CustomDataset is not in the mmseg::dataset registry"
#   "SegLocalVisualizer is not in the mmengine::visualizer registry"
# 另外 mmseg.utils.tokenizer 无条件 import ftfy/regex，而这两个包未被声明为依赖，需手工安装。
custom_imports = dict(
    imports=[
        'mmseg.datasets',
        'mmseg.engine',
        'mmseg.evaluation',
        'mmseg.models',
        'mmseg.structures',
        'mmseg.visualization',
    ],
    allow_failed_imports=False)

# ⚠️ 改成你自己的数据根目录：该目录下需有 images/{train,val} 和 masks3/{train,val}
data_root = 'data'
crop_size = (512, 512)
num_classes = 3
img_suffix = '.jpg'
seg_map_suffix = '.png'

metainfo = dict(
    classes=('background', 'healthy', 'diseased'),
    palette=[[0, 0, 0], [46, 204, 113], [231, 76, 60]])

# ---------------- 模型 ----------------
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size),
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        # CE + Dice 组合：CE 管整体，Dice 直接优化重叠度，小目标收益明显
        # 若以后类别不均衡，给 CE 加 class_weight=[1.0, 1.0, 1.5, 1.5] 即可
        loss_decode=[
            # 若发现"健康"类被压制(前景 mIoU 里 healthy 明显低于 diseased)，
            # 取消注释给少数类加权(当前前景比 健康:病甲 = 1 : 2.53)
            # dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            #      class_weight=[1.0, 1.5, 1.0], loss_name='loss_ce'),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                 loss_name='loss_ce'),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=1.0,
                 loss_name='loss_dice'),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ---------------- 数据 ----------------
# 数据集类用 BaseSegDataset（MMSeg 1.x 已移除 CustomDataset，其功能并入 BaseSegDataset）：
#   ann_file 留空时，load_data_list() 会递归扫描 data_prefix.img_path 下的 *.jpg，
#   并把后缀替换成 .png 去 data_prefix.seg_map_path 找同名掩膜。
# 注意 1.x 用 data_prefix=dict(img_path=..., seg_map_path=...) 而不是 0.x 的 img_dir/ann_dir。
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),  # 统一 512x512，避免小图裁剪报错
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # 小角度旋转：指甲在画面里本来就有各种朝向
    dict(type='RandomRotate', prob=0.5, degree=15, pad_val=0, seg_pad_val=255),
    # 颜色抖动调强：你的图来自不同手机/光源，这是最该增强的维度
    dict(type='PhotoMetricDistortion',
         brightness_delta=40, contrast_range=(0.6, 1.6),
         saturation_range=(0.6, 1.6), hue_delta=20),
    # 随机遮挡，逼模型别只盯局部纹理
    dict(type='RandomCutOut', prob=0.3, n_holes=(1, 3),
         cutout_ratio=[(0.05, 0.15), (0.05, 0.15)],
         fill_in=(0, 0, 0), seg_fill_in=255),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
# TTA: 测试时做多尺度 + 水平翻转，取平均；用 --tta 启用
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='TestTimeAug', transforms=[
        [dict(type='Resize', scale=sz, keep_ratio=False)
         for sz in [(384, 384), (512, 512), (640, 640)]],
        [dict(type='RandomFlip', prob=0., direction='horizontal'),
         dict(type='RandomFlip', prob=1., direction='horizontal')],
        [dict(type='LoadAnnotations')],
        [dict(type='PackSegInputs')],
    ])
]
tta_model = dict(type='SegTTAModel')

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='masks3/train'),
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        metainfo=metainfo,
        reduce_zero_label=False,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='masks3/val'),
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        metainfo=metainfo,
        reduce_zero_label=False,
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator

# ---------------- 训练策略 ----------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
param_scheduler = [
    # 预热占总迭代的 10%（1000 轮 -> 100 轮）；PolyLR 必须 begin < end 且 end == max_iters
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=100),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=100, end=1000, by_epoch=False),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=1000, val_interval=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=50,
        save_best='mIoU', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
randomness = dict(seed=42, deterministic=False)
log_level = 'INFO'
log_processor = dict(by_epoch=False)
# ADE20K 预训练权重（150 类）；加载到 4 类头时会有 size mismatch 警告，属正常
# ADE20K 预训练权重（迁移学习用），需自行下载后放到该路径；置 None 则从零训练
load_from = 'pretrained/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth'
resume = False
work_dir = './work_dirs/segformer-b0_nail_3class'
auto_scale_lr = dict(enable=False, base_batch_size=16)

# dataset settings
dataset_type = 'Box2seg'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomGamma', gamma_range=[80,120]),
    dict(type='Resize', img_scale=(512,512), ratio_range=(1.0,1.0), keep_ratio=False),
    dict(type='Crop_with_Mask', jitter_ratio=0.1, max_range=.5),
    dict(type='Resize', img_scale=(512,512), ratio_range=(1.0,1.0), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


voc_train = dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root='data/voc/',
            img_dir = 'images/',
            ann_dir = 'masks/',
            pipeline=train_pipeline
        )
)

oid_train = dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root='data/oid/',
            img_dir = 'images/',
            ann_dir = 'masks/',
            split = ['train_clean_v2.txt', ],
            pipeline=train_pipeline
        )
)

coco_train = dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root='data/coco/',
            img_dir = 'images/',
            ann_dir = 'masks/train2017/',
            split = ['train_list.txt', ],
            pipeline=train_pipeline
        )
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=[
        voc_train,
        oid_train,
        coco_train,
        ],
    val=dict(
        type=dataset_type,
        data_root='data/test/',
        img_dir='images/',
        ann_dir='masks/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root='data/test/',
        img_dir='images/',
        ann_dir='masks/',
        pipeline=test_pipeline))

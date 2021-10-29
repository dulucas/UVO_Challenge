# dataset settings
dataset_type = 'UVODataset'
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
    dict(type='LoadImageWithBBox'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'],
                meta_keys=('filename', 'ori_filename', \
                           'ori_shape', 'img_shape', 'pad_shape', \
                           'scale_factor', 'flip', 'flip_direction', \
                           'img_norm_cfg', 'pred_bbox', 'pred_score', \
                           'img_shape_before_crop', 'crop_bbox', 'image_id')
                ),
        ])
]

voc_train = dict(
        type=dataset_type,
        data_root='data/uvo/',
        img_dir='YOUR/UVO/PATH/',
        proposal_path='YOUR/PROPOSAL/PATH/xxxx.json',
        gt_path='YOUR/ANNOTATION/PATH/UVO_frame_test.json',
        pipeline=train_pipeline
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=voc_train,
    val=dict(
        type=dataset_type,
        data_root='data/uvo/',
        img_dir='YOUR/UVO/PATH/',
        proposal_path='YOUR/PROPOSAL/PATH/xxxx.json',
        gt_path='YOUR/ANNOTATION/PATH/UVO_frame_test.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root='data/uvo/',
        img_dir='YOUR/UVO/PATH/',
        proposal_path='YOUR/PROPOSAL/PATH/xxxx.json',
        gt_path='YOUR/ANNOTATION/PATH/UVO_frame_test.json',
        pipeline=test_pipeline))

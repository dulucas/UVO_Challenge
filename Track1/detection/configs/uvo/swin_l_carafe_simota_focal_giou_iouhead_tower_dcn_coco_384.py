_base_ = '../rpn/rpn_r50_caffe_fpn_1x_coco.py'
pretrained = 'PATH/TO/YOUR/swin_large_patch4_window12_384_22k.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        load_like_mmseg=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN_CARAFE',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        num_outs=5,
        start_level=0,
        end_level=-1,
        norm_cfg=None,
        act_cfg=None,
        order=('conv', 'norm', 'act'),
        upsample_cfg=dict(
            type='carafe',
            up_kernel=5,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64)),
    rpn_head=dict(
        _delete_=True,
        type='UVORPNHead',
        num_stages=2,
        stages=[
            dict(
                type='StageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[1.0],
                    strides=[4, 8, 16, 32, 64]),
                adapt_cfg=dict(type='dilation', dilation=3),
                dcn_on_last_conv=True,
                bridged_feature=True,
                sampling=False,
                with_cls=False,
                cls_head='cls_head',
                reg_decoded_bbox=False,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.2, 0.2)),
                use_tower_convs=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                ),
            dict(
                type='StageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
                adapt_cfg=dict(type='offset'),
                dcn_on_last_conv=True,
                bridged_feature=False,
                sampling=False,
                with_cls=True,
                cls_head='cls_head',
                reg_decoded_bbox=False,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                use_tower_convs=True,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                )
        ]),
    train_cfg=dict(rpn=[
        dict(
            assigner=dict(type='RPN_SimOTAAssigner', center_radius=.25),
            aux_assigner=dict(type='RPN_SimOTAAssigner', center_radius=.25, candidate_topk=20),
            aux_sampler=None,
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(type='RPN_SimOTAAssigner', center_radius=.25),
            aux_assigner=dict(type='RPN_SimOTAAssigner', center_radius=.25, candidate_topk=20),
            aux_sampler=None,
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
    ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0)))

#tta_flip = True
#tta_scale = [(667, 400), (833, 500), (1000, 600), (1067, 640), (1167, 700),
#             (1333, 800), (1500, 900), (1667, 1000), (1833, 1100),
#             (2000, 1200), (2167, 1300), (2333, 1400), (3000, 1800)]

#scale_ranges = [(96, 10000), (96, 10000), (64, 10000), (64, 10000),
#                (64, 10000), (0, 10000), (0, 10000), (0, 10000), (0, 256),
#                (0, 256), (0, 192), (0, 192), (0, 96)]

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017_clsagn.json',
        img_prefix=data_root + 'train2017/',
        #pipeline=train_pipeline,
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017_clsagn.json',
        img_prefix=data_root + 'val2017/',
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017_clsagn.json',
        img_prefix=data_root + 'val2017/',
        ))
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=12)

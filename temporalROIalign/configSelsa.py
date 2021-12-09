model = dict(
    detector=dict(
        type='FasterRCNN',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            strides=(1, 2, 2, 1),
            dilations=(1, 1, 1, 2),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
        neck=dict(
            type='ChannelMapper',
            in_channels=[2048],
            out_channels=512,
            kernel_size=3),
        rpn_head=dict(
            type='RPNHead',
            in_channels=512,
            feat_channels=512,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[4, 8, 16, 32],
                ratios=[0.5, 1.0, 2.0],
                strides=[16]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=0.1111111111111111,
                loss_weight=1.0)),
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]),
            bbox_head=dict(
                type='SelsaBBoxHead',
                in_channels=512,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.2, 0.2, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                num_shared_fcs=2,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16))),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=6000,
                max_per_img=600,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=6000,
                max_per_img=300,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0001,
                nms=dict(type='nms', iou_threshold=0.7),
                max_per_img=100))),
    type='SELSA')
dataset_type = 'ImagenetVIDDataset'
data_root = 'dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=[
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            ann_file='trackData/Video0/train0.json',
            img_prefix='trackData/Video0/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=9,
                filter_key_img=True,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video1/train1.json',
            img_prefix='trackData/Video1/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video2/train2.json',
            img_prefix='trackData/Video2/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video3/train3.json',
            img_prefix='trackData/Video3/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),

        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video4/train4.json',
            img_prefix='trackData/Video4/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video6/train6.json',
            img_prefix='trackData/Video6/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video7/train7.json',
            img_prefix='trackData/Video7/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video8/train8.json',
            img_prefix='trackData/Video8/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),

        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video9/train9.json',
            img_prefix='trackData/Video9/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video10/train10.json',
            img_prefix='trackData/Video10/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video11/train11.json',
            img_prefix='trackData/Video11/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video12/train12.json',
            img_prefix='trackData/Video12/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video13/train13.json',
            img_prefix='trackData/Video13/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video14/train14.json',
            img_prefix='trackData/Video14/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video15/train15.json',
            img_prefix='trackData/Video15/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video16/train16.json',
            img_prefix='trackData/Video16/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video17/train17.json',
            img_prefix='trackData/Video17/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),
        dict(
            classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='trackData/Video18/train18.json',
            img_prefix='trackData/Video18/combined',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ]),


            
    ],
    val=dict(
        classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
        type='ImagenetVIDDataset',
        ann_file='trackData/Video5/train5.json',
        img_prefix='trackData/Video5/combined',
        ref_img_sampler=dict(
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride'),
        pipeline=[
            dict(type='LoadMultiImagesFromFile'),
            dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
            dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
            dict(
                type='SeqNormalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='SeqPad', size_divisor=16),
            dict(
                type='VideoCollect',
                keys=['img'],
                meta_keys=('num_left_ref_imgs', 'frame_stride')),
            dict(type='ConcatVideoReferences'),
            dict(type='MultiImagesToTensor', ref_prefix='ref'),
            dict(type='ToList')
        ],
        test_mode=True),
    test=dict(
        classes=('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'),
        type='ImagenetVIDDataset',
        ann_file='trackData/Video5/train5.json',
        img_prefix='trackData/Video5/combined',
        ref_img_sampler=dict(
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride'),
        pipeline=[
            dict(type='LoadMultiImagesFromFile'),
            dict(type='SeqResize', img_scale=(1024, 128), keep_ratio=True),
            dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
            dict(
                type='SeqNormalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='SeqPad', size_divisor=16),
            dict(
                type='VideoCollect',
                keys=['img'],
                meta_keys=('num_left_ref_imgs', 'frame_stride')),
            dict(type='ConcatVideoReferences'),
            dict(type='MultiImagesToTensor', ref_prefix='ref'),
            dict(type='ToList')
        ],
        test_mode=True))
optimizer = dict(type='SGD', lr=0.001, momentum=0.90, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
checkpoint_config = dict(interval=2)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[6, 10])
total_epochs = 20
evaluation = dict(metric=['bbox'], interval=1)
work_dir = './work_dirs/selsa_faster_rcnn_r50_dc5_1x_imagenetvid'
gpu_ids = range(0, 1)

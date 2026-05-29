_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

map_classes = [
    'drivable_area', 'ped_crossing', 'walkway',
    'stop_line', 'carpark_area', 'divider',
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=True,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 200
bev_w_ = 200
queue_length = 1  # VJepa features carry their own temporal info

group_detr = 3

# V-JEPA token grid (must match the h5 extraction resolution)
# h5 stores (6, T*vjepa_h_*vjepa_w_, 768) per sample
# For 368x656 input with 16x16 patches: H=23, W=41
vjepa_h_ = 28
vjepa_w_ = 50
vjepa_feat_h = 448
vjepa_feat_w = 800

vjepa_h5_path = '/mnt/vilab/scratch/masha/vjepa_cache/train_feat_vitb_4x448x800.h5'
train_ann_file = '/mnt/vilab/scratch/masha/nuscenes_trainval/nuscenes_infos_temporal_train_with_map_200.pkl'
val_ann_file   = '/mnt/vilab/scratch/masha/nuscenes_trainval/nuscenes_infos_temporal_val_with_map_200.pkl'

model = dict(
    type='BEVFormerVJepa',
    use_grid_mask=False,
    video_test_mode=False,
    pretrained=None,
    vjepa_in_dim=768,
    vjepa_out_dim=_dim_,
    vjepa_h=vjepa_h_,
    vjepa_w=vjepa_w_,
    vjepa_temporal_reduce='last',
    vjepa_adapter_hidden_dim=512,
    vjepa_adapter_num_blocks=4,
    vjepa_gate_hidden_dim=256,
    img_backbone=None,
    img_neck=None,
    pts_bbox_head=dict(
        type='BEVFormerHead_GroupDETR',
        group_detr=group_detr,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='GroupMultiheadAttention',
                            group=group_detr,
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    map_seg_head=dict(
        type='MapSegHead',
        in_channels=_dim_,
        num_classes=len(map_classes),
        loss_seg=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=[
                0.50,   # drivable_area  — common, ~30-40% of BEV
                0.75,   # ped_crossing   — rare
                0.65,   # walkway        — moderate
                0.90,   # stop_line      — very rare, thin lines
                0.65,   # carpark_area   — moderate
                0.90,   # divider        — very rare, thin lines
            ],
            loss_weight=10.0,
        ),
    ),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
data_root = '/mnt/vilab/scratch/masha/nuscenes_trainval/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadVJepaFeaturesFromH5',
        h5_path=vjepa_h5_path,
        group='vitb',
        img_h=vjepa_feat_h,
        img_w=vjepa_feat_w,
        orig_w=1600,
        orig_h=900,
    ),

    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='LoadMapMaskFromNpz', classes=map_classes),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='VJepaFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_masks_bev']),
]

test_pipeline = [
    dict(
        type='LoadVJepaFeaturesFromH5',
        h5_path=vjepa_h5_path,
        group='vitb',
        img_w=vjepa_feat_w,
        img_h=vjepa_feat_h,
        orig_w=1600,
        orig_h=900
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(800, 448),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='VJepaFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=2000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-3)
total_epochs = 30

evaluation = dict(
    interval=3,
    metric='bbox',
    pipeline=test_pipeline,
    start=1,
    save_best='pts_bbox_NuScenes/NDS',
    rule='greater'
)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='bevformer_map',
                name='vjepa_448x800_bev200_map_aux_resadapter4_last_segw10',
                dir='/mnt/vilab/scratch/masha/wandb',
                config=dict(
                    model='BEVFormerVJepa',
                    features='V-JEPA cached',
                    vjepa_resolution='448x800',
                    vjepa_h=vjepa_h_,
                    vjepa_w=vjepa_w_,
                    temporal_reduce='last',
                    adapter='residual adapter',
                    adapter_hidden_dim=512,
                    adapter_num_blocks=4,
                    spatial_num_points=8,
                    bev_h=bev_h_,
                    bev_w=bev_w_,
                    map_classes=map_classes,
                    map_loss_weight=10.0,
                    group_detr=group_detr,
                    samples_per_gpu=4,
                    workers_per_gpu=2,
                    queue_length=queue_length,
                    detection_head='BEVFormerHead_GroupDETR',
                    map_head='MapSegHead',
                )
            )
        ),
    ]
)
checkpoint_config = dict(interval=2)

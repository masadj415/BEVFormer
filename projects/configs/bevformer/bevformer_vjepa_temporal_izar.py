# V-JEPA BEVFormer config
# BEV = 200 x 200
# Encoder layers = 6
# Decoder layers = 6
# V-JEPA feature level = 1
# Adapter = 768 -> 512 -> 512 -> 256 with 3x3 convs
# Temporal fusion = token-wise gated fusion over cached V-JEPA temporal slices

_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True
)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2

# V-JEPA gives one feature map level only.
_num_levels_ = 1

# BEVFormer-base-like BEV resolution.
bev_h_ = 200
bev_w_ = 200

# Keep queue_length = 1 because temporal fusion is inside cached V-JEPA slices.
# The current BEVFormerVJepa forward_train does not use prev_bev.
queue_length = 1

model = dict(
    type='BEVFormerVJepa',
    use_grid_mask=False,
    video_test_mode=True,
    pretrained=None,

    # V-JEPA cached feature settings.
    vjepa_in_dim=768,
    vjepa_out_dim=_dim_,
    vjepa_h=28,
    vjepa_w=50,

    # New temporal fusion.
    vjepa_temporal_reduce='last',
    vjepa_adapter_num_blocks=4,
    vjepa_adapter_hidden_dim=512,
    vjepa_gate_hidden_dim=256,

    # Track 1: Ego trajectory head.
    # Predicts 6 future ego waypoints (3 s at 0.5 s/step) in current ego frame.
    # Initialized from canbus (speed/yaw prior) + 2-layer BEV cross-attention.
    # loss_weight here is the initial/floor value; AuxLossWarmupHook ramps it
    # linearly to 0.5 between epoch 9 and 15.
    ego_trajectory_head=dict(
        type='EgoTrajectoryHead',
        embed_dims=_dim_,
        num_waypoints=6,
        canbus_dim=18,
        num_decoder_layers=2,
        num_heads=8,
        dropout=0.1,
        loss_weight=0.02,
    ),

    # Track 2: Per-agent motion head.
    # Piggybacks on the 900 DETR detection queries (no new queries).
    # GT is real future positions from the motion pkl (not velocity extrapolation).
    # Only moving agents (|v| > 0.5 m/s) contribute to the loss.
    # loss_weight here is the initial/floor value; AuxLossWarmupHook ramps it
    # linearly to 0.25 between epoch 9 and 15.
    motion_head=dict(
        type='MotionHead',
        embed_dims=_dim_,
        num_waypoints=6,
        future_dt=0.5,
        loss_weight=0.02,
    ),

    img_backbone=None,
    img_neck=None,

    pts_bbox_head=dict(
        type='BEVFormerHead',
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
                            num_levels=1
                        ),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                im2col_step=24,
                                num_levels=_num_levels_
                            ),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn', 'norm',
                        'cross_attn', 'norm',
                        'ffn', 'norm'
                    )
                )
            ),

            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            im2col_step=24,
                            num_levels=1
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn', 'norm',
                        'cross_attn', 'norm',
                        'ffn', 'norm'
                    )
                )
            )
        ),

        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10
        ),

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
            loss_weight=2.0
        ),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=0.25
        ),
        loss_iou=dict(
            type='GIoULoss',
            loss_weight=0.0
        )
    ),

    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range
            )
        )
    )
)

dataset_type = 'CustomNuScenesDataset'
data_root = '/scratch/izar/mduric/nuscenes_trainval/'
motion_ann_file = '/transfer/NaTaMaPa/nuscenes_metadata/nuscenes_infos_temporal_train_with_map_200_motion.pkl'
val_ann_file = '/transfer/NaTaMaPa/nuscenes_metadata/nuscenes_infos_temporal_val_with_map_200_motion.pkl'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadVJepaFeaturesFromH5',
        h5_path='/transfer/NaTaMaPa/train_feat_vitb_4x448x800.h5',
        group='vitb',
        img_w=800,
        img_h=448,
        orig_w=1600,
        orig_h=900
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    # Trajectory-aware versions keep gt_fut_traj / gt_fut_traj_mask in sync
    # with gt_bboxes_3d through each filter step.
    dict(
        type='ObjectRangeFilterWithTraj',
        point_cloud_range=point_cloud_range
    ),
    dict(
        type='ObjectNameFilterWithTraj',
        classes=class_names
    ),
    dict(
        type='VJepaFormatBundle3D',
        class_names=class_names
    ),
    dict(
        type='CustomCollect3D',
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img',
              'gt_future_ego', 'gt_fut_traj', 'gt_fut_traj_mask']
    )
]

test_pipeline = [
    dict(
        type='LoadVJepaFeaturesFromH5',
        h5_path='/transfer/NaTaMaPa/train_feat_vitb_4x448x800.h5',
        group='vitb',
        img_w=800,
        img_h=448,
        orig_w=1600,
        orig_h=900
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(800, 448),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='VJepaFormatBundle3D',
                class_names=class_names,
                with_label=False
            ),
            dict(
                type='CustomCollect3D',
                keys=['img']
            )
        ]
    )
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,

    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=motion_ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        box_type_3d='LiDAR'
    ),

    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1
    ),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality
    ),

    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01
)

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2)
)

lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

total_epochs = 30

# Gradually ramp auxiliary loss weights from 0.02 → target over epoch 9-15.
# Prevents auxiliary gradients from conflicting with detection before the
# shared BEV/query representations have stabilised.
custom_hooks = [
    dict(
        type='AuxLossWarmupHook',
        start_epoch=9,
        end_epoch=12,
        ego_target=0.5,
        motion_target=0.25,
        min_weight=0.02,
    )
]

runner = dict(
    type='EpochBasedRunner',
    max_epochs=total_epochs
)

evaluation = dict(
    interval=3,
    metric='bbox',
    pipeline=test_pipeline,
    save_best='pts_bbox_NuScenes/NDS',
    rule='greater'
)


log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='bevformer-vjepa',
                entity='lord-of-the-strings',
                name='vjepa_bev_448x800_temporal_bev_4_with_our_best_adapter',
                dir='/scratch/izar/tlphan/wandb',
                config=dict(
                    model='BEVFormerVJepa',
                    features='V-JEPA cached',
                    temporal_reduce='last',
                    adapter='4 blocks, 512 hidden dim',
                    gate_hidden_dim=256,
                    samples_per_gpu=4,
                    workers_per_gpu=2,
                    queue_length=queue_length,
                    bev_h=bev_h_,
                    bev_w=bev_w_,
                    encoder_layers=6,
                    decoder_layers=6,
                    num_levels=_num_levels_
                )
            )
        )
    ]
)

checkpoint_config = dict(
    interval=2
)
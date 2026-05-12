_base_ = ['./bevformer_small_map.py']

# Finetune only the ego trajectory head.
# Load from the 30-epoch joint checkpoint, freeze everything else.

# Lower LR — head is small, BEV features are frozen
optimizer = dict(
    type='AdamW',
    lr=3e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            # Freeze all params not in ego_traj_head
            'pts_bbox_head':  dict(lr_mult=0.0, decay_mult=0.0),
            'map_seg_head':   dict(lr_mult=0.0, decay_mult=0.0),
            'vjepa_adapter':  dict(lr_mult=0.0, decay_mult=0.0),
            'vjepa_temporal_gate': dict(lr_mult=0.0, decay_mult=0.0),
        }
    )
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-3)

total_epochs = 10
evaluation = dict(interval=2, metric='ego', start=2)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=2)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='bevformer_map',
                 name='ego_finetune',
                 config=dict(model='BEVFormerVJepa', task='ego_trajectory'),
             )),
    ])

resume_from = None
load_from = '/transfer/NaTaMaPa/bev_small_map/latest.pth'
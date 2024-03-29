# Data cfg
dataset_type = 'ImageNetDataset'
# data_root = '/mnt/c/Users/sherw/OneDrive/Desktop/SSL_Proj/Implementation/Dataset/Data_Temp/test_sample'  # data path  ####################
# data_root = '/mnt/c/Users/sherw/OneDrive/Desktop/SSL_Proj/Implementation/Dataset/Data_Temp/test_sample/'
# data_root = '/mnt/e/Data/test/Backgrounds'
# data_root = '/mnt/e/Data/test/PackedGenerated(OneListPer)'
data_root = '/mnt/c/Users/sherw/OneDrive/Desktop/PackedGenerated(OneListPer)'


base_scale = (256, 256)
fore_scale = (128, 255)
# img_norm_cfg = dict(                                            ###########
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)         ######

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)


# preprocess_pipeline = dict(                               #####
#     type='CopyAndPaste',
#     feed_bytes=False,
#     base_scale=base_scale,
#     ratio=(0.5, 2.0),
#     w_range=fore_scale,
#     h_range=fore_scale,
# )

train_pipeline = [
    # dict(type='Resize', img_scale=(500, 500), keep_ratio=True),
    
    # dict(type='Resize', img_scale=(500, 500), keep_ratio=True),         #########
    # dict(type='PixelAugPil', to_rgb=True),                              ########
    dict(type='RandomFlip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),                                  #########
    dict(type='DefaultFormatBundle', to_tensor=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# data = dict(
#     # samples_per_gpu=32,                                                                                           ########################
#     samples_per_gpu=2,
#     # workers_per_gpu=4,                                                                                               #######################
#     workers_per_gpu=4,
#     train=dict(
#         type=dataset_type,
#         # ann_file='/mnt/c/Users/sherw/OneDrive/Desktop/SSL_Proj/Implementation/Dataset/Data_Temp/data_ann.txt',  # anno file  ##############
#         ann_file='/mnt/c/Users/sherw/OneDrive/Desktop/SSL_Proj/Implementation/Dataset/Data_Temp/data_ann.txt',  # anno file
#         img_prefix=data_root,
#         preprocess=preprocess_pipeline,
#         pipeline=train_pipeline))


data = dict(
    # samples_per_gpu=32,                                                                                           ########################
    samples_per_gpu=4,
    # workers_per_gpu=4,                                                                                               #######################
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        # ann_file='/mnt/c/Users/sherw/OneDrive/Desktop/SSL_Proj/Implementation/Dataset/Data_Temp/data_ann.txt',  # anno file  ##############
        # ann_file='/mnt/c/Users/sherw/OneDrive/Desktop/SSL_Proj/Implementation/Dataset/Data_Temp/data_ann.txt',  # anno file
        # ann_file='/mnt/e/Data/test/data_ann.txt',  # anno file
        # ann_file='/mnt/e/Data/test/data_ann.txt',  # anno file
        ann_file='/mnt/c/Users/sherw/OneDrive/Desktop/data_ann.txt',
        img_prefix=data_root,
        pipeline=train_pipeline))

# Model Cfg
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='InsLocFPN',
    pretrained=None,
    # pool_with_gt=[False, True],  # for query, key encoder respectively        ##############
    pool_with_gt=[True, True],                                                  ##############
    drop_rpn_k=True,
    shuffle_data=['img', 'bbox'],
    num_levels=5,
    level_loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
    box_replaced_with_gt=[True, True, True,
                          True, True],  # Means the box aug is only applied on P5    #??????
    # momentum_cfg=dict(dim=128, K=65536, m=0.999, T=0.2),                                                      ######################
    momentum_cfg=dict(dim=128, K=4, m=0.999, T=0.2),
    backbone=dict(
        type='ResNet',
        # depth=50,
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        normal_init=True,
        zero_init_residual=False),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_cfg=norm_cfg,
        num_outs=5),                                ######
        # num_outs=4),                               #????????
    rpn_head=dict(                                  #???????
        type='AnchorAugHead',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8, 16, 32],
            # scales=[8],
            ratios=[0.5, 0.75, 1.0, 1.5, 2.0],
            # strides=[4, 8, 16, 32, 64]),        
            strides=[16]),
    ),
    roi_head=dict(
        type='MomentumRoIPool',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=0),
            # num_extractor=4,
            num_extractor=5,
            out_channels=256,
            finest_scale=56,
            featmap_strides=[4, 8, 16, 32, 64]),
        bbox_head=dict(
            type='ConvFCBBoxInsClsHead',
            num_shared_convs=4,
            num_shared_fcs=2,
            fc_out_channels=1024,
            with_avg_pool=False,
            norm_cfg=norm_cfg,
            roi_feat_size=7,
            in_channels=256,
            final_out_channel=128,
        )))
train_cfg = dict(
    rpn=None,
    rpn_proposal=dict(
        nms_across_levels=False,
        generate_from_single_level=2,
        iou_thr=0.5,
        iou_nms=True,
        nms_pre=4000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=None,
    fp16 = True,
)
test_cfg = dict(rpn=None, rcnn=None)

# Training cfg
optimizer = dict(
    type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001, nesterov=True)

optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnealing', min_lr=0.0, by_epoch=True)
# total_epochs = 200                                                                                ###########################################
total_epochs = 40
checkpoint_config = dict(interval=1)
#log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])                               ############################
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)

# cudnn_benchmark = True          ########################

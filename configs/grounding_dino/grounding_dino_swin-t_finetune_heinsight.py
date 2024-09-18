_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

data_root = '/home/junru/heinsight5/Dataset/crop/Merged/'
#data_root = '/home/junru/heinsight5/Dataset/ambiguous_YOLO_dataset/train/'

class_name = ('Hetero', 'Empty', 'Residue', 'Solid', 'Homo',)
# class_name = ('this is prompts.')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),])



model = dict(bbox_head=dict(num_classes=num_classes))

# training
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations_coco/instances_train.json',
        data_prefix=dict(img='train/images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations_coco/instances_test.json',
        data_prefix=dict(img='test/images/')))


# train_dataloader = dict(
#     batch_size=1,
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='aligned_labels_coco/instances_train.json',
#         data_prefix=dict(img='images/case1_2/')))

# val_dataloader = dict(
#     dataset=dict(
#         metainfo=metainfo,
#         data_root=data_root,
#         ann_file='aligned_labels_case2_coco/instances_train.json',
#         data_prefix=dict(img='images/case2')))


test_dataloader = val_dataloader


## evaluation
val_evaluator = dict(ann_file=data_root + 'annotations_coco/instances_test.json')
# val_evaluator = dict(ann_file=data_root + 'aligned_labels_case2_coco/instances_train.json')
test_evaluator = val_evaluator

max_epoch = 30

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=10))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=16)

# load_from = '/home/junru/heinsight5/mmdetection/heinsightamb_work_dir/best_coco_bbox_mAP_epoch_25.pth'
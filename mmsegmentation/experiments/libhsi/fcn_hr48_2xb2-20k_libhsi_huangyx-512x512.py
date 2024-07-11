_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/libhsi.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375])
dataset = dict(data_prefix=dict(seg_map_path='train/gt-huangyx'))
train_dataloader = dict(dataset=dataset)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384],
        channels=sum([48, 96, 192, 384]),
        num_classes=27))
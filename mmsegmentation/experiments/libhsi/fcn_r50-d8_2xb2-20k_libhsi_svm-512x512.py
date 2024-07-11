_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/libhsi.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
)
dataset = dict(data_prefix=dict(seg_map_path='train/generated-svm'))
train_dataloader = dict(dataset=dataset)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=27),
    auxiliary_head=dict(num_classes=27))

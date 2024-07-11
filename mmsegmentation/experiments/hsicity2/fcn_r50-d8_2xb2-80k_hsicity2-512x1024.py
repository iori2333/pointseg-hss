_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/hsicity2.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    loss_scale=512.)
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

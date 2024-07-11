export CUDA_VISIBLE_DEVICES=2

python dist_test.py --model ResNet --cube_width 11 --state_dict hsicity2_work_dirs/ResNet-w11/epoch_15.pth --data_root data/HSICityV2 --work_dir hsicity2_work_dirs
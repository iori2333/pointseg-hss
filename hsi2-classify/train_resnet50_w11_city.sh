export CUDA_VISIBLE_DEVICES=2

python train.py --model ResNet --cube_width 11 --data_root data/HSICityV2 --work_dir hsicity2_work_dirs
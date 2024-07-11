export CUDA_VISIBLE_DEVICES=3

python train_libhsi.py --model ResNet --cube_width 11 --data_root data/LIB-HSI --work_dir libhsi_work_dirs
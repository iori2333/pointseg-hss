export CUDA_VISIBLE_DEVICES=3

python test_libhsi.py --model ResNet --cube_width 11 --state_dict libhsi_work_dirs/ResNet-w11/epoch_15.pth --data_root data/LIB-HSI --work_dir libhsi_work_dirs
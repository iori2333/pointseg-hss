export CUDA_VISIBLE_DEVICES=3

python test_libhsi.py --model RSSAN --cube_width 17 --state_dict libhsi_work_dirs/RSSAN-w17/epoch_15.pth --data_root data/LIB-HSI --work_dir libhsi_work_dirs

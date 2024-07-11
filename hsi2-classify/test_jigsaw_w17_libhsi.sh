export CUDA_VISIBLE_DEVICES=2

python test_libhsi.py --model JigSawHSI --cube_width 17 --state_dict libhsi_work_dirs/JigSawHSI-w17/epoch_15.pth --data_root data/LIB-HSI --work_dir libhsi_work_dirs

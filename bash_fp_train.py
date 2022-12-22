import os
cmd = 'python train_bair.py --dataset bair --model vgg --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --lr 5e-5 --n_future 10 --channels 3 --data_root /home/zhengchang/Datasets/datasets/bair/bair_robot_pushing_dataset_v0/  --log_dir results/bair/'

print(cmd)
os.system(cmd)
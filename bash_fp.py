import os

cmd_load_model = 'scp zchang@10.129.167.212:/backup1/zhengchang/code/svg_stau/logs_lp_DataAugmented/bair/model=vgg64x64-rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0001-g_dim=128-z_dim=64-last_frame_skip=0-beta=0.0001000/continued/continued/model_last.pth ./pretrained_models/fp/model.pth'
# os.system(cmd_load_model)
# 19.522871445327223 0.846664139204319 280.5842 273.77032
# 100 20.023917344982916 0.865606094862801 0.05431380015111894
cmd_test = 'python test_bair.py --model_path checkpoints/bair/model.pth --log_dir results/bair/ --data_root /home/zhengchang/Datasets/datasets/bair/bair_robot_pushing_dataset_v0/ --batch_size 32 --nsample 100 --rand_num 5'

os.system(cmd_test)

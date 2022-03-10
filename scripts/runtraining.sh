CUDA_VISIBLE_DEVICES=0 python train_crowd.py -c configs/crowd_configs/base.yaml -i 2000 -e ../builds/crowd-v5/crowd.x86_64 -n car_vel_new_highent -d CartesianVelocity -o Absolute -p crowdai-v5

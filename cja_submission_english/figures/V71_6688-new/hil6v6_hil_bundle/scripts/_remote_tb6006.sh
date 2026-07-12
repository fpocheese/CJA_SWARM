#/bin/bash install.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlgpu
cd ~/000000GSY_mutiUAV/swarm_attack_v2
exec tensorboard --logdir outputs/results/fov_penetration/mappo --port 6006 --host 0.0.0.0 --reload_interval 30

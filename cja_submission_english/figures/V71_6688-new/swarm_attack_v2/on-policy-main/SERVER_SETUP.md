# 服务器部署指南 - MARL Algorithm Comparison

## 1. 环境配置

### 方式一：使用 Conda (推荐)
```bash
# 创建新环境
conda create -n rlgpu python=3.8 -y
conda activate rlgpu

# 安装 PyTorch (根据服务器CUDA版本选择)
# CUDA 11.8:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121
# CPU only:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements_rlgpu.txt
```

### 方式二：使用 pip + venv
```bash
python -m venv rlgpu_env
source rlgpu_env/bin/activate
pip install -r requirements_rlgpu.txt
```

## 2. 代码结构

```
on-policy-main/
├── onpolicy/
│   ├── envs/mpe/                    # MPE环境
│   │   ├── scenarios/
│   │   │   ├── simple_spread_v3.py  # 简化的spread环境 (3 agents)
│   │   │   └── simple_world_comm.py # FighterWorld环境 (20 defense + 8 attack)
│   │   └── ...
│   ├── scripts/
│   │   ├── train_spread_v3.py       # Spread环境训练脚本
│   │   ├── train_world_comm.py      # WorldComm环境训练脚本
│   │   └── results/                 # 训练结果保存目录
│   └── algorithms/                  # 算法实现
├── requirements_rlgpu.txt           # 依赖文件
└── SERVER_SETUP.md                  # 本文件
```

## 3. 运行训练

### Simple Spread V3 环境 (快速测试，推荐先运行这个)
```bash
cd onpolicy/scripts

# 完整训练 (1000 ep × 6 algo × 3 seeds, 约20分钟)
python train_spread_v3.py --num_episodes 1000

# 快速测试 (100 ep × 1 algo × 1 seed)
python train_spread_v3.py --num_episodes 100 --seeds 1 --algorithms MAPPO

# 只生成图表 (训练完成后)
python train_spread_v3.py --plot_only
```

### Simple World Comm 环境 (复杂环境，20个agents)
```bash
cd onpolicy/scripts

# 完整训练 (500 ep × 6 algo × 3 seeds, 约3小时)
python train_world_comm.py --num_episodes 500

# 快速测试
python train_world_comm.py --num_episodes 100 --seeds 1 --algorithms MAPPO
```

## 4. 算法说明

| 算法 | 类型 | 特点 |
|------|------|------|
| MAPPO | On-policy | 集中式Critic，标准PPO |
| Advanced-MAPPO | On-policy | 带Residual Block的改进版 |
| IPPO | On-policy | 独立Critic |
| IA2C | On-policy | 独立Actor-Critic，无PPO clip |
| IQL | Off-policy | 独立Q-learning |
| MADDPG | Off-policy | 集中式训练分布式执行 |

## 5. 关键发现

**重要**: 在单环境设置下，`ppo_epoch=1` 是关键！更高的值会导致严重过拟合。

## 6. 结果输出

训练完成后，结果保存在:
- `onpolicy/scripts/results/simple_spread_v3/` - Spread环境结果
- `onpolicy/scripts/results/simple_world_comm/` - WorldComm环境结果

输出文件:
- `*_rewards.npy` - 每个算法每个seed的训练奖励曲线
- `comparison_*.png/pdf` - 对比图
- `final_performance_bar.png/pdf` - 最终性能条形图

## 7. 后台运行

```bash
# 使用 nohup 后台运行
cd onpolicy/scripts
nohup python -u train_spread_v3.py --num_episodes 1000 > train.log 2>&1 &

# 查看进度
tail -f train.log

# 查看进程
ps aux | grep train_spread
```

# 🚀 快速入门指南 - 改进版MAPPO

## 5分钟快速开始

### 步骤1: 修改base_runner.py以支持改进版

首先需要修改base_runner来支持新的训练器：

```bash
# 编辑 onpolicy/runner/shared/base_runner.py
# 在 __init__ 方法中，找到创建trainer的部分
```

将以下代码：
```python
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
```

修改为：
```python
# 根据配置选择算法版本
if hasattr(all_args, 'use_advanced_algo') and all_args.use_advanced_algo:
    from onpolicy.algorithms.r_mappo.r_mappo_advanced import R_MAPPO_Advanced as TrainAlgo
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_advanced import R_MAPPOPolicy_Advanced as Policy
else:
    from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
```

### 步骤2: 运行训练

```bash
cd /home/uav/00gao_xueshu/togsy_2025/0620septimedone/on-policy-main/onpolicy/scripts

# 给脚本执行权限
chmod +x train_mpe_advanced.sh

# 运行训练
./train_mpe_advanced.sh
```

### 步骤3: 监控训练

训练过程中会看到：
```
==================================================
Scenario: simple_world_comm | Algo: rmappo
Episode: 100/1000 | Steps: 60000/10000000
FPS: 1234
Average Episode Rewards: 15.67
KL Divergence: 0.018234
KL Coefficient: 0.125
Value Warmup: False
==================================================
```

## 📊 训练建议时间线

### 阶段1: 价值函数热身 (Episodes 1-100)
- **现象**: 只有Critic更新，Actor保持不变
- **目的**: 让价值估计先稳定
- **监控**: `Value Warmup: True`

### 阶段2: 联合训练 (Episodes 101-500)
- **现象**: Actor和Critic同时更新
- **目的**: 策略开始学习和优化
- **监控**: KL散度应在0.01-0.03之间

### 阶段3: 精细调优 (Episodes 501+)
- **现象**: 学习率逐渐降低（如果使用调度器）
- **目的**: 策略稳定收敛
- **监控**: 回报曲线趋于平稳

## 🎛️ 关键超参数调整指南

### 场景1: 训练不收敛

```bash
# 问题: 回报一直很低，没有明显提升

# 解决方案1: 增加热身时间
--warmup_episodes 200

# 解决方案2: 降低学习率
--lr 1e-4
--critic_lr 1e-4

# 解决方案3: 增加网络容量
--hidden_size 512
```

### 场景2: 训练不稳定（波动大）

```bash
# 问题: 回报曲线上下波动剧烈

# 解决方案1: 加强clip
--clip_param 0.1
--dual_clip_param 2.0

# 解决方案2: 梯度裁剪
--max_grad_norm 5.0

# 解决方案3: 使用Huber loss
--use_huber_loss True
--huber_delta 10.0
```

### 场景3: 收敛太慢

```bash
# 问题: 训练很久才有效果

# 解决方案1: 增加探索
--entropy_coef 0.05

# 解决方案2: 提高学习率
--lr 1e-3
--critic_lr 1e-3

# 解决方案3: 减少PPO epoch
--ppo_epoch 5
```

### 场景4: 性能达到瓶颈

```bash
# 问题: 收敛后性能不够好

# 解决方案1: 增大网络
--hidden_size 512
--layer_N 3

# 解决方案2: 更细致的训练
--ppo_epoch 15
--num_mini_batch 2

# 解决方案3: 调整折扣因子
--gamma 0.995  # 更看重长期回报
```

## 📈 与原版对比测试

运行对比测试：

```bash
# 1. 先运行原版MAPPO（用于对比）
cd scripts
./train_mpe.sh  # 使用原版脚本

# 2. 运行改进版
./train_mpe_advanced.sh

# 3. 对比结果
python compare_algorithms.py \
  --original_dir ../results/MPE/simple_world_comm/rmappo/your_exp_name \
  --advanced_dir ../results/MPE/simple_world_comm/advanced_mappo/your_exp_name \
  --save_dir ./comparison_results
```

## 🔧 自定义你的场景

### 修改启动脚本

编辑 `train_mpe_advanced.sh`:

```bash
# 1. 修改场景名称（改成你的场景）
scenario="your_custom_scenario"

# 2. 修改智能体数量
num_agents=10  # 拦截方UAV数量 + 目标方UAV数量

# 3. 修改episode长度
episode_length=1000  # 根据你的任务时长

# 4. 根据任务特点选择参数

# 如果需要强协同:
--use_attention True
--hidden_size 256

# 如果动作空间很大:
--entropy_coef 0.05  # 增加探索

# 如果状态空间很大:
--hidden_size 512
--layer_N 3
```

## 💾 保存和加载模型

### 保存
模型会自动保存在：
```
results/MPE/{scenario_name}/advanced_mappo/{exp_name}/models/
├── actor.pt
└── critic.pt
```

### 加载和继续训练

```python
# 在train_mpe_advanced.py中添加
if args.model_dir is not None:
    runner.restore()
```

然后运行：
```bash
./train_mpe_advanced.sh --model_dir path/to/saved/models
```

## 🐛 常见错误排查

### 错误1: ImportError

```
ImportError: No module named 'onpolicy.algorithms.r_mappo.r_mappo_advanced'
```

**解决**: 确保所有新文件都已创建，路径正确

### 错误2: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**解决**: 减少并行环境或网络大小
```bash
--n_rollout_threads 16  # 从32减到16
--hidden_size 128       # 从256减到128
```

### 错误3: Runner创建失败

```
ImportError: cannot import name 'MPERunner'
```

**解决**: 确保使用正确的Runner
```python
from onpolicy.runner.shared.mpe_runner_advanced import MPERunner as Runner
```

## 📊 性能评估

### 评估指标

1. **Average Episode Rewards**: 最重要指标，越高越好
2. **KL Divergence**: 应在0.01-0.03之间
3. **Policy Loss**: 应逐渐下降
4. **Value Loss**: 应逐渐下降
5. **Entropy**: 初期高（探索），后期低（收敛）

### 判断训练是否成功

✅ **训练成功的标志**:
- 回报曲线持续上升
- KL散度稳定在target附近
- Loss曲线下降并稳定
- FPS稳定

❌ **需要调整的标志**:
- 回报长时间不上升
- KL散度持续很高（>0.1）
- Loss曲线不下降
- 训练崩溃（NaN）

## 🎯 针对固定翼拦截的特殊建议

### 观察空间设计
```python
# 建议包含:
- 自身状态: 位置、速度、姿态
- 队友状态: 相对位置、速度
- 目标状态: 相对位置、速度
- 任务信息: 剩余时间、成功率等
```

### 动作空间设计
```python
# 建议:
- 连续动作: 俯仰角、偏航角、油门
- 离散动作: 特技动作（翻滚、跃升等）
```

### 奖励函数设计
```python
# 推荐结构:
reward = (
    + 100 * intercept_success      # 成功拦截奖励
    - 1 * distance_to_target       # 距离惩罚
    + 10 * team_coordination       # 协同奖励
    - 0.1 * control_effort         # 能耗惩罚
    - 100 * collision              # 碰撞惩罚
)
```

## 📚 下一步

1. ✅ 完成基础训练
2. ✅ 对比原版性能
3. ⬜ 调优超参数
4. ⬜ 在真实场景测试
5. ⬜ 部署到实际系统

## 🆘 需要帮助？

1. 查看 `README_ADVANCED.md` 了解详细信息
2. 检查训练日志中的错误信息
3. 调整超参数后重试
4. 确保环境和奖励设计合理

---

**祝训练顺利！** 🎉

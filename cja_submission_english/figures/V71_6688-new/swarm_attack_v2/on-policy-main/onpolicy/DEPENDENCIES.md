# 改进版MAPPO环境配置说明

## 依赖包

改进版MAPPO需要以下Python包：

### 基础依赖
```
torch>=1.8.0
numpy>=1.19.0
gym>=0.21.0
```

### 训练相关
```
tensorboard
wandb  # 可选，用于云端日志
```

### 环境相关
```
absl-py
```

### 可视化
```
matplotlib
imageio
```

## 安装方法

### 方法1: 使用requirements.txt（推荐）

如果项目根目录有requirements.txt：
```bash
pip install -r requirements.txt
```

### 方法2: 手动安装

```bash
# 安装PyTorch（根据你的CUDA版本选择）
# CPU版本
pip install torch torchvision torchaudio

# GPU版本（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy gym tensorboard matplotlib imageio absl-py
```

### 方法3: 使用conda（推荐）

```bash
# 创建新环境
conda create -n mappo_advanced python=3.8

# 激活环境
conda activate mappo_advanced

# 安装PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
conda install numpy gym tensorboard matplotlib imageio -c conda-forge
pip install absl-py wandb
```

## 验证安装

安装完成后，运行验证脚本：

```bash
cd /home/uav/00gao_xueshu/togsy_2025/0620septimedone/on-policy-main/onpolicy
python scripts/verify_installation.py
```

如果看到：
```
✓ 所有测试通过！代码可以正常使用。
```

说明环境配置成功！

## 常见问题

### Q1: 没有GPU怎么办？

A: 可以在启动脚本中去掉 `--cuda` 参数：
```bash
# 编辑 train_mpe_advanced.sh
# 删除或注释最后的 --cuda
```

### Q2: CUDA版本不匹配

A: 检查CUDA版本并安装对应的PyTorch：
```bash
# 检查CUDA版本
nvidia-smi

# 根据版本安装，例如CUDA 11.7
pip install torch --index-url https://download.pytorch.org/whl/cu117
```

### Q3: absl包冲突

A: 尝试重新安装：
```bash
pip uninstall absl-py
pip install absl-py
```

## 下一步

环境配置完成后：

1. 检查代码完整性
```bash
ls -la algorithms/r_mappo/algorithm/r_actor_critic_advanced.py
ls -la algorithms/r_mappo/r_mappo_advanced.py
```

2. 修改训练参数
```bash
vim scripts/train_mpe_advanced.sh
```

3. 开始训练
```bash
cd scripts
./train_mpe_advanced.sh
```

## 最小依赖版本

如果遇到版本冲突，以下是测试过的最小版本：

```
python>=3.7
torch>=1.8.0
numpy>=1.19.0
gym>=0.21.0
tensorboard>=2.0.0
matplotlib>=3.3.0
imageio>=2.9.0
absl-py>=0.12.0
```

## 推荐配置

为了获得最佳性能：

```
python==3.8
torch==1.13.0
numpy==1.23.0
gym==0.21.0
```

---

如果遇到其他问题，请检查：
1. Python版本是否正确
2. 是否在正确的虚拟环境中
3. 包版本是否兼容

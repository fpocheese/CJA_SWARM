# V71 五个成功击中案例数据包 — 数据验收报告

**生成时间**: 2026-05-11  
**数据来源**: 远端 (a2rl@172.20.10.4) `/tmp/v71_figs/` 搜索、补录、本地同步

## 📊 数据概览

| 工况 | 环境种子 | 策略种子 | 命中步数 | 最近距离 | 文件数 | 状态 |
|-----|--------|--------|--------|--------|-------|------|
| caseA_seed50015_torch1 | 50015 | 1 | 4683 | 4.90m | 3 | ✓ |
| caseB_seed50042_torch1 | 50042 | 1 | 5014 | 4.73m | 3 | ✓ |
| caseC_seed50034_torch7 | 50034 | 7 | 4707 | 4.89m | 3 | ✓ |
| caseD_seed50042_torch7 | 50042 | 7 | 4888 | 4.58m | 3 | ✓ |
| caseE_seed50015_torch8 | 50015 | 8 | 4944 | 4.60m | 3 | ✓ |

## 📁 每个工况的数据包结构

### 每个工况目录包含三个数据文件：

1. **trajectory_data.npz** (~876-997 KB)
   - 轨迹数据：4个进攻者 + 4个防守者 + HVT 的完整时间序列
   - 关键字段：
     - `off_*`: 进攻者位置、速度、导航指令、存活状态、击中状态
     - `def_*`: 防守者位置、速度、锁定模式、目标分配
     - `hvt_*`: HVT 位置、进攻者-HVT 距离
     - `hit_count`: 累计击中数
   - 时间长度：4684~4945 步

2. **summary.json** (~1-2 KB)
   - 元数据：env_seed, torch_seed, hit_step, best_dist (如适用)
   - 示例：
     ```json
     {
       "env_seed": 50015,
       "torch_seed": 1,
       "hit_step": 4683,
       "best_dist": null
     }
     ```

3. **game_data.npz** (~873-1020 KB)
   - 游戏论 theoretica 指标和动态数据
   - 关键字段：
     - `hvt_rho`: HVT极坐标距离 (shape: (4 agents, T steps))
     - `hvt_P_hit`: HVT 击中概率
     - `hvt_closing`: HVT 距离闭合速率
     - `pen_N_eff`, `pen_P_pen`: 渗透指标
     - `esc_Gamma_mean`, `esc_Xi_mean`, `esc_E_esc`: 逃避指标
     - `def_lmode`, `def_ltgt`: 防守者锁定模式与目标
     - `decoy_*`: 诱饵角色与压力

## ✅ 数据一致性验证

### 三个数据包的闭环同步检查：
- ✓ 所有 5 个工况都有完整的三个数据包
- ✓ `hit_step` 值在 trajectory_data 和 game_data 的时间范围内
- ✓ 在 hit_step 处，所有进攻者的距离值 (off_d_hvt 与 hvt_rho) 在数值上一致
- ✓ 最近的进攻者在命中时距离均 < 5m（命中阈值 = 5m）
- ✓ 所有进攻者活跃状态记录一致

### 数据质量指标：
- 命中进攻者最近距离范围：4.58~4.90 m（均低于 5m 阈值）
- 每个工况 HVT 到进攻者的 4 个距离通道都被正确记录
- 游戏指标（hvt_rho, hvt_P_hit 等）覆盖完整时间序列

## 🗑️ 清理操作

**删除的重复旧工况**：
- ~~caseC_seed50042_torch7~~ (timestamp: 2026-05-11 17:00)
- ~~caseD_seed50015_torch8~~ (timestamp: 2026-05-11 17:00)
- ~~caseE_seed50034_torch7~~ (timestamp: 2026-05-11 17:00)

**保留的新工况**（生成时间 2026-05-11 22:04~22:08）：
- caseA_seed50015_torch1 ✓
- caseB_seed50042_torch1 ✓
- caseC_seed50034_torch7 ✓
- caseD_seed50042_torch7 ✓
- caseE_seed50015_torch8 ✓

## 📄 验收报告

详细的数据验收报告已保存至 `DATA_VALIDATION_REPORT.json`。

---

**数据状态**：✓ **已验收** — 所有工况数据完整、同步、可用于论文绘图。

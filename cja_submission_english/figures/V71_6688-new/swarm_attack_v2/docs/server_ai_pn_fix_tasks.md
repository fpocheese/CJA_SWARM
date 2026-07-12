# 服务器 AI 操作任务：PN 制导检查与修复

> 工作目录：仓库根目录  
> 不得修改环境物理参数（`dt`、`max_steps`、`hit_hvt_range`、`collision_kill_range`、`an_pitch_max`、`an_yaw_max`、`v_min`、`v_max` 等）。

---

## 任务 1：检查并修复拦截器 PN 制导

文件：`envs/fov_penetration/policies_interceptor.py`

### 问题 1A：`_pn_guidance_3d` 内部闭合速度是否实时计算？

找到 `_pn_guidance_3d` 方法，检查它用于 PN 指令的 `v_closing`（或 `Vc`）是否在方法内部由当前相对位置和速度重新计算，还是直接读取了 `self.closing_speed`（上一步缓存值）。

- 如果是直接读取 `self.closing_speed` 而没有重新计算 → **这是 bug，需要修复**
- 正确做法：在 `_pn_guidance_3d` 内部用当前 `(dx, dy, dz)` 和相对速度 `(dvx, dvy, dvz)` 重新计算：

```python
v_closing = -(dx * dvx + dy * dvy + dz * dvz) / r
self.closing_speed = v_closing  # 同步更新缓存
```

然后用这个 `v_closing` 参与 PN 指令计算。

### 问题 1B：PN 指令是否使用向量投影公式？

检查 `_pn_guidance_3d` 的侧向加速度指令计算方式。

**错误方式**（标量 LOS 角速率，需要修改）：
```python
an_yaw_cmd   = N * v_closing * los_rate_az
an_pitch_cmd = N * v_closing * los_rate_el
```

**正确方式**（向量 PN 投影，不受飞行路径角影响）：
```python
range_vec    = np.array([dx, dy, dz])
relative_vel = np.array([dvx, dvy, dvz])   # v_target - v_interceptor
velocity_vec = np.array([vx_i, vy_i, vz_i])

los_omega_vec = np.cross(range_vec, relative_vel) / max(r * r, 1e-6)
velocity_axis = velocity_vec / max(np.linalg.norm(velocity_vec), 1.0)
accel_cmd = N * max(v_closing, 0.0) * np.cross(los_omega_vec, velocity_axis)

yaw_axis   = np.array([-np.sin(intc.heading), np.cos(intc.heading), 0.0])
pitch_axis = np.array([
    -np.sin(intc.gamma) * np.cos(intc.heading),
    -np.sin(intc.gamma) * np.sin(intc.heading),
    np.cos(intc.gamma),
])
an_yaw_cmd   = float(np.dot(accel_cmd, yaw_axis))
an_pitch_cmd = float(np.dot(accel_cmd, pitch_axis)) + G * np.cos(intc.gamma)
```

如果当前是标量 LOS 角速率方式，改为向量投影方式。

### 问题 1C：饱和保护是否完整？

确认 `_pn_guidance_3d` 末尾对加速度指令做了飞机包线饱和保护：

```python
params = intc.params
ax_cmd       = np.clip(ax_cmd,       params["ax_min"],        params["ax_max"])
an_pitch_cmd = np.clip(an_pitch_cmd, -params["an_pitch_max"], params["an_pitch_max"])
an_yaw_cmd   = np.clip(an_yaw_cmd,   -params["an_yaw_max"],   params["an_yaw_max"])
```

如果没有，补充上述保护。

---

## 任务 2：检查进攻飞行器末端 PN 制导

文件：`scripts/terminal_pn_action_wrapper.py`

### 问题 2A：文件是否存在？

- 不存在 → 跳过本任务，在报告中注明。
- 存在 → 继续以下检查。

### 问题 2B：闭合速度计算是否正确？

找到 `_terminal_pn_action`（或等效的进攻方 PN 指令计算函数），检查闭合速度的计算。目标 HVT 是静止目标，正确写法：

```python
range_vec    = target_pos - own_pos   # 指向 HVT 的向量
relative_vel = -own_vel               # = v_target - v_own = 0 - v_own
closing_speed = -np.dot(range_vec, relative_vel) / range_norm
             # = np.dot(range_vec, own_vel) / range_norm
```

如果写法数学等价 → 正确，无需修改。

### 问题 2C：PN 指令是否使用向量投影法？

要求与任务 1B 相同：必须用向量叉积法计算 `los_omega_vec`，再 `cross(los_omega_vec, velocity_axis)` 得到加速度向量，再投影到飞机控制轴。不能直接用标量 LOS 角速率。

### 问题 2D：后半球退化处理是否存在？

当飞机已经飞过目标（`bearing_err > 90°` 或 `closing_speed ≤ 0`）时，PN 公式会发散，需要切换到纯追踪制导。检查是否有如下逻辑：

```python
if abs(bearing_err) > math.pi / 2 or closing_speed <= 0.0:
    # 纯追踪模式
    an_yaw_cmd   = np.clip(4.0 * bearing_err, -5.0, 5.0) * G
    los_el       = math.atan2(range_vec[2], math.hypot(range_vec[0], range_vec[1]))
    pitch_err    = wrap_angle(los_el - own_gamma)
    an_pitch_cmd = np.clip(3.0 * pitch_err, -4.0, 4.0) * G + G * math.cos(own_gamma)
```

如果没有此退化处理，补充上述逻辑。

### 问题 2E：`terminal_only` 默认值是否为 `True`？

确认 `__init__` 签名：
```python
def __init__(self, env, gain=3.0, max_action=0.8, terminal_only=True, ...):
```

`terminal_only=True` 的含义：仅当 phase flag 触发（末端制导阶段）时才用 PN 接管动作，否则保留 MAPPO 策略动作。如果默认值是 `False`，改为 `True`。

---

## 任务 3：统一所有 PN 导引系数 N = 3

搜索全仓库中所有 `pn_nav_gain`、`pn_gain`、`terminal_pn_gain`、`FOV_TERMINAL_PN_GAIN` 的赋值，全部改为 3 或 3.0。

具体检查点：

| 文件 | 字段 | 目标值 |
|------|------|--------|
| `envs/fov_penetration/config.py` | `"pn_nav_gain"` 字典项 | `3` |
| `envs/fov_penetration/config.py` | `analytic_priors` 里的 `"pn_nav_gain"` | `3.0` |
| `scripts/terminal_pn_action_wrapper.py` | `__init__` 的 `gain` 默认参数 | `3.0` |
| `scripts/train_fov_penetration_mappo.py` | `--terminal_pn_gain` 的 `default=` | `3.0` |
| `scripts/train_fov_penetration_mappo.py` | `os.environ.get('FOV_TERMINAL_PN_GAIN', ...)` | `'3.0'` |
| `scripts/run_v69_hybrid_terminal_pn.sh` | `export FOV_TERMINAL_PN_GAIN=` | `3.0` |
| `scripts/run_v69_hybrid_terminal_pn.sh` | `--terminal_pn_gain` 参数值 | `3.0` |
| `scripts/run_v69_hourly_monitor.sh` | `FOV_TERMINAL_PN_GAIN=` | `3.0` |
| `scripts/eval_v69_collect.py` | `--pn-gain` 的 `default=` | `3.0` |
| `scripts/eval_v69_monte_carlo.py` | `--pn-gain` 的 `default=` | `3.0` |
| `scripts/eval_v69_monte_carlo_batch.py` | `--pn-gain` 的 `default=` | `3.0` |
| `scripts/eval_v69_monte_carlo_vec.py` | `--pn-gain` 的 `default=` | `3.0` |

规则：
- 文件不存在 → 跳过该行，报告中注明
- 字段值已经是 3 或 3.0 → 无需修改，报告中注明"已是正确值"
- **不得修改其他任何数值**

---

## 最终报告格式

完成上述三项任务后，请提供以下报告：

**任务 1 报告**
- `_pn_guidance_3d` 发现的问题（closing_speed、向量投影、饱和保护）
- 修改的行号及内容（原 → 新）；若无问题注明"无需修改"

**任务 2 报告**
- 文件是否存在
- 各子项（2B/2C/2D/2E）是否有问题，修改内容；若无问题注明"无需修改"

**任务 3 报告**
- 各字段：已是正确值 / 修改（原值 → 新值） / 文件不存在（跳过）

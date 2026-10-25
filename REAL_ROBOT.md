# 真机测试指南

本文件说明在这台 CR5A 上跑轨迹需要准备什么、怎么验证、出问题怎么退。

**先读最后一节「安全边界」。**

---

## 1. 你需要提供 / 确认的信息

跑真机前必须确认这 6 项。带 ❗ 的是必须由你现场提供，我无法代填。

| # | 项目 | 当前值 | 说明 |
|---|---|---|---|
| 1 ❗ | 控制器 IP | `192.168.5.1` | 默认值。若改过，用 `--ip` 传入 |
| 2 ❗ | 牙模实际位姿 | `(-34, -46, 463)mm`, `rpy (90,0,90)°` | **这是我搜索出的可达位置，不是你现场的坐标。** 换成手眼标定值 |
| 3 ❗ | 工具 TCP 标定 | `(0, 120, 150)mm` | 按你的规范填的。真机实测一次四点法确认 |
| 4 | 控制器固件版本 | ≥ V4.4.0.0 | 低于此版本 ServoJ 参数格式不同 |
| 5 | 控制柜网络模式 | TCP/IP 远程 | DobotStudio Pro 必须断开（29999 单客户端） |
| 6 | 关节软限位 | `±180°`（j3 为 `±160°`） | URDF 里是 ±360°，我按物理实际收紧了。**请对照手册核实** |

第 2、3 项最关键。位姿错了会撞，TCP 错了轨迹整体偏移。

---

## 2. 首次连接检查（不动机械臂）

```bash
cd D:\CompanyProject\DrobotGeneratPath

# 只做检查，不发任何运动指令
python scripts/real_test.py --trajectory out/arch_gen.json --ip 192.168.5.1 --check-only
```

预期输出：

```
  OK   all 111 waypoints inside joint limits
  INFO max joint step 1.4 deg
  INFO RobotMode: 0,{...};
  INFO current joints deg: [...]
  OK   start pose XX.X deg from current
  OK   feedback stream alive (N packets)
READY
```

任何 `FAIL` 都不要继续。常见原因：

- `cannot connect to :29999` → IP 错、网线没通、或控制柜不在远程模式
- `no packets from port 30004` → 反馈流未开，监控会失效，不建议继续
- `start pose 距离过大` → 机械臂当前姿态离轨迹起点太远，先手动移到安全位姿附近

---

## 3. 空载低速试跑

**第一次务必空载**：刷头不接触牙模，牙模可以先拿走。

```bash
# 5% 速度，会要求输入 YES 确认
python scripts/real_test.py --trajectory out/arch_gen.json --ip 192.168.5.1 --speed 5
```

观察要点：

1. 机械臂先 `MovJ` 到起点，这段应该缓慢、平滑
2. 进入 ServoJ 后，终端每 50 个点打印 `follow_err`
3. `follow_err` 应稳定在 1–3°，超过 8° 会自动停止

全程手放急停。`Ctrl+C` 会立即发 `Stop()`。

### 分级提速

```
5%  →  确认方向和范围正确
10% →  确认轨迹贴合牙面
20% →  正常测试
```

不要跳过 5% 直接跑高速。

---

## 4. 带负载测试

空载通过后，装上刷头、放回牙模：

```bash
# 10% 速度，带负载
python scripts/real_test.py --trajectory out/arch_gen.json --ip 192.168.5.1 --speed 10
```

`standoff_m` 决定刷头离牙面的距离，默认 2mm。首次带载建议临时改成 5mm（不接触），确认路径正确后再降到 2mm。

改法：`config/tasks/arch_walk.yaml` 里的 `standoff_m: 0.002` → `0.005`，重新 `plan`。

---

## 5. 执行日志

每次运行都会写 CSV：

```
out/arch_gen_realtest.csv
```

列含义：`t_s, i, cmd_j1..6, fb_j1..6, follow_err_deg`。

### 怎么读

```python
import pandas as pd
d = pd.read_csv("out/arch_gen_realtest.csv")
print(d["follow_err_deg"].describe())
# 看哪一段跟踪误差大
import matplotlib.pyplot as plt
plt.plot(d["t_s"], d["follow_err_deg"])
plt.xlabel("time [s]"); plt.ylabel("follow error [deg]")
plt.show()
```

跟踪误差持续偏大通常意味着：

- 速度设太高（降低 `--speed`）
- 刷头接触力过大（增大 `standoff_m`）
- 控制器负载接近上限（检查是否加了额外工具）

---

## 6. 模块接口

如果要在自己的代码里调用：

```python
from dobot_pathgen.backends.dobot_tcp import DashboardClient, FeedbackReader
from dobot_pathgen.config import Config
from dobot_pathgen.planning import Trajectory

cfg = Config.load("config/robot.yaml")
traj = Trajectory.load("out/arch_gen.json")

dash = DashboardClient("192.168.5.1")   # 29999 命令端口
dash.connect()
dash.clear_error()
dash.enable()
dash.speed_factor(10)

reader = FeedbackReader("192.168.5.1", 30004)   # 30004 反馈流
reader.start()

for w in traj.waypoints:
    deg = [math.degrees(v) for v in w.positions]
    dash.servo_j(deg, t=0.02)

reader.stop()
dash.close()
```

**不要**在真机上直接调用 `servo_j` 而不做跟踪误差监控——那是 `scripts/real_test.py` 存在的原因。

---

## 7. 安全边界

**这个项目验证过什么：** 运动学规划、IK 可达性、关节限位、工具与机械臂的几何间隙。全部在离线环境完成，44 个测试覆盖。

**没验证过什么：**

1. **Isaac Sim 从未成功启动。** 本机 pip 版 Kit 在 `app ready` 后段错误，疑与 Todesk 虚拟显示驱动有关。`isaac_sim.py` 的代码写好了但没跑通。
2. **真机从未运行。** 没有任何一条轨迹在物理机械臂上执行过。
3. **接触力完全没建模。** 不做刷毛压缩、水射流、力的任何仿真。
4. **牙模位姿是搜索值，不是标定值。** `(-34, -46, 463)mm` 是我为了同时满足可达性和 10mm 碰撞间隙搜出来的，不是你现场的真实位置。

**已知风险：**

- `joint5 = 88.7°` 处接近腕部奇异。起始位姿就在这个点附近，真机可能报奇异警告。
- 刷棒长度 192mm，关节空间的小偏差会在大半径上放大。
- 301 个路径点连续 ServoJ，控制器队列深度有限，网络抖动可能导致不平滑。

**必须做的：**

- 首次运行空载、5% 速度、手放急停
- 先 `--check-only`，再看 `follow_err`，再提速
- 牙模位姿换成你自己标定的值后，重新 `plan` 并看 `proximity gate`

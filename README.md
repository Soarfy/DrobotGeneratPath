# DobotPathGeneration — CR5A 牙模清洁轨迹 Demo

最小可跑通的闭环：**牙模网格 → 表面轨迹 → IK 关节轨迹 → 间隙校验 → Isaac Sim 回放 → 真机执行**。

用一条 JSON 轨迹文件贯穿仿真和真机，两边只换执行后端，规划结果完全一致。

当前配置对应你给的下颌牙模 `LowerJawScan.ply` 和 200 mm 直棒刷头（直径 40 mm，纯 XYZ 偏置 12 cm 前 + 16 cm 下，无旋转）。

## 当前 Demo 做了什么

- 从牙模 PLY 里按角度扫描提取牙弓脊线，或按区域做蛇形覆盖。
- 在表面上生成路径，按法向偏置固定间距，工具 +Z 指向牙面。
- 用 CR5A 官方 URDF 做数值 IK，带关节限位、连续性检查和时间参数化。
- 校验关节限位/速度/加速度，并检查**棒身与机械臂本体的间隙**。
- 输出 `out/*.json` 轨迹文件和校验报告。
- 同一个文件可以在 Isaac Sim 里回放，也可以通过 ROS2 或直连 TCP 在真机上执行。

这是**运动学/几何 Demo**：不模拟刷毛变形、接触力或水射流。力控是后续独立的一步。

## 目录结构

```
config/
  robot.yaml                    机器人、工具、场景、规划参数
  tasks/demo_occlusal_patch.yaml 示例任务（牙合面一块区域）
assets/
  robot/cr5a/                   CR5A 官方 URDF + STL（来自 Dobot ROS2 仓库）
  robot/cr5a_with_tool/         机器人 + 刷头组合 URDF（Isaac Sim 用这个导入）
  target/lower_jaw.ply          下颌牙模原始网格（毫米，133k 顶点）
  target/lower_jaw.stl          转成米的 STL，给 Isaac Sim
src/dobot_pathgen/
  se3.py                        SE(3)、RPY、四元数
  urdf_chain.py                 URDF 解析、正运动学、雅可比
  ik.py                         阻尼最小二乘 IK、多起点、连续性优先
  mesh_io.py                    PLY/STL 读写、法向
  pathgen.py                    区域选择、蛇形覆盖、牙弓脊线、位姿生成、加密
  planning.py                   任务定义、轨迹编译、限位校验
  calibration.py                TCP / 牙模位姿标定文件解析
  collision.py                  工具-机械臂、工具-工件间隙检查
  config.py                     YAML 配置
  cli.py                        命令行入口
  backends/
    isaac_sim.py                Isaac Sim 回放
    ros2_backend.py             通过官方 ROS2 驱动执行
    dobot_tcp.py                直连 TCP 执行（不需要 ROS）
    replay.py                   HTML/PLY 离线预览
scripts/
  make_assets.py                生成刷头网格、组合 URDF、STL
  preflight.py                  真机运行前检查（含间隙校验）
launch_isaac_sim.py             用 Isaac Sim 自带 Python 启动
tests/test_core.py              核心单元测试
```

## 安装

需要 Python 3.9+（Windows 上直接用你现有的 Anaconda 即可）。

```bash
cd D:\CompanyProject\DrobotGeneratPath
pip install -e .
pip install pytest
```

如果只想跑规划、不装包：

```bash
set PYTHONPATH=src
```

## 真机测试

真机执行用独立模块，带安全检查：

```bash
# 1. 静态检查，不动机械臂
python scripts/real_test.py --trajectory out/arch_gen.json --ip 192.168.5.1 --check-only

# 2. 空载低速试跑（5%），会要求输入 YES
python scripts/real_test.py --trajectory out/arch_gen.json --ip 192.168.5.1 --speed 5
```

它会检查关节限位、跟踪误差（超过阈值自动停）、记录命令与反馈的 CSV。**完整准备清单和风险说明见 [REAL_ROBOT.md](REAL_ROBOT.md)。**

> 注意：这个项目的轨迹**从未在真机上运行过**，Isaac Sim 也未能启动。所有验证都在运动学层面完成。

## 怎么运行（先看这个）

**一条命令看仿真效果：**

```bat
D:\CompanyProject\DrobotGeneratPath\run_demo.bat
```

会自动规划轨迹、跑校验、渲染视频，并用浏览器打开可交互的 3D 查看器。完整说明见 [RUN.md](RUN.md)。

已经验证过的输出：

```
waypoints       : 186
duration        : 50.1 s
IK failures     : 0
max joint step  : 3.5 deg
validation      : PASS
tool-arm gap    : 68.0 mm
proximity gate  : PASS
```

| 产物 | 路径 |
|---|---|
| 轨迹 JSON | `out/arch_lower_jaw.json` |
| 校验报告 | `out/arch_lower_jaw.meta.json` |
| 离线视频 | `out/vtk_preview/robot_path.mp4` |
| 交互查看器 | `out/viewer/viewer.html`（需本地 HTTP，用 `open_viewer.bat` 启动） |

## 快速开始

**1. 生成轨迹**

两种轨迹模式：

```bash
# 沿牙弓走一遍（磨牙 -> 前牙 -> 磨牙）
dobot-pathgen plan --task config/tasks/arch_walk.yaml --out out/arch_walk.json

# 或者只扫某一块区域（蛇形覆盖）
dobot-pathgen plan --task config/tasks/demo_occlusal_patch.yaml --out out/trajectory.json
```

`arch_walk.yaml` 的预期输出（186 点全部可达，棒身离机械臂 68 mm）：

```
task            : arch_walk
pattern         : arch
surface points  : 33
waypoints       : 186
duration        : 50.1 s
path length     : 0.182 m
IK failures     : 0
max joint step  : 3.5 deg
velocity usage  : 9%
validation      : PASS
tool-arm gap    : 68.0 mm (min at wp 59, Link4)
tool-workpiece  : 0.1 mm (min at wp 151)
proximity gate  : PASS
```

`demo_occlusal_patch.yaml`（单颗前牙区域，蛇形覆盖）：

```
waypoints       : 164
duration        : 38.9 s
IK failures     : 0
max joint step  : 8.8 deg
validation      : PASS
tool-arm gap    : 69.1 mm
proximity gate  : PASS
```

**2. 离线预览（不需要 Isaac Sim）**

```bash
dobot-pathgen preview --trajectory out/trajectory.json --out out/preview
```

用浏览器打开 `out/preview/path_preview.html`，红色是工具路径，灰色是牙模点云。

**3. Isaac Sim 回放**

用 Isaac Sim 自带的 Python，不要用系统 Python：

```powershell
C:\isaac-sim\python.bat launch_isaac_sim.py --trajectory out/trajectory.json
```

无窗口运行：

```powershell
C:\isaac-sim\python.bat launch_isaac_sim.py --trajectory out/trajectory.json --headless
```

**4. 真机执行**

先做静态检查（不动机械臂）：

```bash
python scripts/preflight.py --ip 192.168.5.1
```

再以 10% 速度跑：

```bash
dobot-pathgen execute --trajectory out/trajectory.json --backend direct --ip 192.168.5.1 --speed-ratio 10
```

或者走官方 ROS2 驱动：

```bash
export IP_address=192.168.5.1
export DOBOT_TYPE=cr5
ros2 launch dobot_bringup_v4 dobot_bringup_ros2.launch.py   # 终端 1
ros2 launch dobot_moveit dobot_moveit.launch.py             # 终端 2
dobot-pathgen execute --trajectory out/trajectory.json --backend ros2
```

## 配置文件说明

### 标定文件（TCP + 牙模位姿）

先看一下标定文件解析出来是什么：

```bash
dobot-pathgen calib --file 你的标定结果.yaml
```

直接写回配置：

```bash
dobot-pathgen calib --file 你的标定结果.yaml --apply
```

支持的格式（`config/calibration.example.yaml` 是模板）：

```yaml
# 工具尖端在 Link6 坐标系下的位姿
tcp:
  xyz_mm: [0, 0, 160]      # 或 xyz_m / matrix / pose
  rpy_deg: [0, 0, 0]

# 牙模坐标系在机器人基坐标系下的位姿（手眼标定结果）
model_to_base:
  xyz_mm: [450, 0, 250]
  rpy_deg: [90, 0, 90]
```

也接受 JSON、纯文本（16 个数按行展开成 4×4，或 6 个数 `x y z rx ry rz`）。
`config/robot.yaml` 里加一段就能自动加载：

```yaml
calibration:
  file: config/my_calibration.yaml
```

### 机器人/场景配置

`config/robot.yaml` 里最关键的是这几项：

```yaml
scene:
  # 牙模坐标系 → 机器人基坐标系。正式使用时换成手眼标定得到的值。
  model_to_base:
    translation_m: [0.45, 0.0, 0.25]
    rpy_deg: [90.0, 0.0, 90.0]

tool:
  # 200 mm 直棒刷头，直径 40 mm，纯 XYZ 偏置、无旋转。
  # 坐标系约定：tool +Z = 接近方向，tool +X = 前进方向。
  # 所以 12 cm 前 + 16 cm 下 = [0.12, 0, 0.16]。
  tcp_offset_m: [0.12, 0.0, 0.16]
  tcp_rpy_deg: [0.0, 0.0, 0.0]
  radius_m: 0.02

robot:
  home_q_rad: [-0.258, -1.836, 1.145, 1.468, -2.470, -0.374]
```

`home_q_rad` 同时是 IK 的种子姿态和运行前的准备位姿。它是在避开腕部奇异（joint5 接近 ±90°）的前提下选的，改动它会影响整条轨迹的 IK 分支。

`model_to_base` 的摆位不能随便改。工件放在 0.45 m 时机械臂折得太紧，200 mm 的棒身会蹭到上臂（实测 0.3 mm）；放到 0.65 m / 0.30 m 后同一条路径有 68 mm 间隙。

### 任务配置

`config/tasks/arch_walk.yaml`（沿牙弓）：

```yaml
pattern: arch
arch:
  up_direction: [0.0, 1.0, 0.0]   # 牙合面朝上的方向
  height_band_m: 0.012            # 只考虑靠近牙合面的顶点
  u_range: null                   # 可选：只走某一段
  lateral_offset_m: 0.0           # 侧向偏移（颊侧/舌侧）
  trim_fraction: 0.0
spacing_m: 0.005                  # 相邻接触点间距
standoff_m: 0.002                 # 离面距离
max_tilt_deg: 40.0                # 工具相对牙合面的最大倾角
tilt_reference: [0.0, 1.0, 0.0]
```

`max_tilt_deg` 是沿牙弓能跑通的关键。牙弓一圈的表面法向会转约 180°，如果严格贴合法向，腕部会在中途跨过一次 IK 分支翻转，导致大量点不可达。限制倾角后 204 个点全部可达、最大关节步 4.6°。

`config/tasks/demo_occlusal_patch.yaml`（局部区域）：

```yaml
region:              # 三选一：bbox / sphere / normal_cone
  mode: bbox
  min_m: [-0.030, 0.005, -0.028]
  max_m: [-0.010, 0.040, -0.010]
spacing_m: 0.003     # 路径点间距
row_spacing_m: 0.003 # 相邻行程间距
standoff_m: 0.002    # 工具尖端离面距离
speed_m_s: 0.01
```

## 真机安全边界

首次运行时请确认：

1. **控制柜已切到 TCP/IP 远程模式**，DobotStudio Pro 已断开（29999 端口只允许一个客户端）。
2. `scripts/preflight.py` 全部通过。
3. `--speed-ratio` 从 10 开始，确认轨迹方向正确后再提高。
4. 手放在急停上；真机首次运行建议空载、不接触牙模。
5. 牙模和工具夹具已经锁紧。

`--dry-run` 只打印前几条 ServoJ 指令，不发送任何东西：

```bash
dobot-pathgen execute --backend direct --dry-run
```

## 已知限制

- **只做运动学验证**：Isaac Sim 里回放的是关节轨迹，不模拟刷毛接触力。
- **绝对精度**：轨迹按名义 URDF 计算，真机绝对定位误差约 0.5–1 mm。Demo 中的站立距离是 2 mm，刷毛可以吸收这部分误差；后续要加力控。
- **反馈是 10 Hz**：`dobot_tcp.py` 用 30004 做监控，不做闭环。
- **4 cm 刷头比单颗牙宽**：贴面时刷头边缘会贴近相邻牙面（实测 0.02–0.3 mm）。这是真实几何，不是算错；如果需要边缘也留间隙，加大 `standoff_m` 或换小直径工具。
- **栅格模式只验证了单牙区域**：跨中线的宽区域会在蛇形换行处出现 IK 分支翻转。全牙弓请用 `arch_walk.yaml`。
- **倾角上限是工程折中**：`max_tilt_deg` 限制工具偏离牙合面法向的角度。严格贴合法向会让腕部在牙弓中途翻转，导致大量点不可达。
- **Isaac Sim 导入未实测**：这台机器没装 Isaac Sim，`backends/isaac_sim.py` 只做了导入路径的版本兼容，没有真正跑过。

## 下一步

1. 把 `model_to_base` 换成手眼标定的实际结果（现在 0.65/0/0.30 是为了避开自碰撞挑的安全摆位，不是你现场的真实坐标）。
2. 用真实刷头尺寸核对 `tcp_offset_m` 和 `radius_m`，并称重更新 URDF 里的工具质量。
3. 全牙弓分 2–3 段（左/前/右），每段单独抬升换位，避免长距离连续绕行。
4. 加机械柔顺末端或六维力传感器，做接触力闭环。
5. Isaac Sim 里加相机，做合成数据或视觉验证。

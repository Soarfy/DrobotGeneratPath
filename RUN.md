# 怎么运行

三条路径，从"立刻能看"到"接真机"。**第 1 条已经在你这台机器上跑通了。**

---

## 1. 一键跑仿真（已验证可用）

双击或在 cmd 里执行：

```bat
D:\CompanyProject\DrobotGeneratPath\run_demo.bat
```

它会依次做四件事：

1. 规划轨迹（沿牙弓 186 个点）
2. 跑校验（关节限位、速度、加速度、工具-机械臂间隙）
3. 渲染一段离线视频 → `out\vtk_preview\robot_path.mp4`
4. 生成可交互的 3D 查看器并用浏览器打开 → `out\viewer\viewer.html`

浏览器里可以：

- 点「播放」看机械臂沿牙弓走完整条轨迹
- 拖滑动条跳到任意时刻
- 鼠标拖动旋转视角、滚轮缩放
- 点「循环」切换是否重复播放

界面左上角实时显示时间、路点编号；右上角是图例（机械臂 / 刷棒 / 牙模 / 轨迹）。

### 其它用法

run_demo.bat raster      :: 改用单牙区域的蛇形覆盖任务
run_demo.bat isaac       :: 额外尝试启动 Isaac Sim（见第 3 节）

### 手动分步执行

如果想单独跑某一步：

```bash
cd D:\CompanyProject\DrobotGeneratPath

# 规划
python -m dobot_pathgen.cli plan --task config/tasks/arch_walk.yaml --out out/arch_lower_jaw.json

# 校验（不接机器人）
python scripts/preflight.py --trajectory out/arch_lower_jaw.json --skip-network

# 离线视频（需要 pyvista，用 Uclean 环境）
D:\SoftWare\Anaconda\envs\Uclean\python.exe scripts\render_vtk.py --trajectory out/arch_lower_jaw.json --frames 80

# 交互查看器
python scripts/export_viewer.py --trajectory out/arch_lower_jaw.json --out out/viewer

**查看器需要本地 HTTP 服务**（浏览器不允许 `file://` 读 JSON）：

cd out\viewer
python -m http.server 8765

然后打开 http://127.0.0.1:8765/viewer.html


## 1.5 刷头与安全位姿规范（当前配置）

| 项目 | 值 | 来源 |
|---|---|---|
| 安全位姿（起点） | `(264.89, -285.19, 491.07)mm / (-179.77, -1.35, -145.91)°` | `BrushTooth.cpp:215-221` |
| 对应关节角 | `[-32.4, -55.2, 51.9, -87.0, 88.7, 23.5]°` | 用 120/150 刷头反解 |
| 刷头长轴 | 平行 Link6 **+Y** | 按你的规范重建网格 |
| 刷头末端偏移 | `(0, +120, +150)mm`，X 分量为 0 | 同上 |
| 在安全位姿下 | 尖端在法兰下方 150.4mm、前方 96.9mm | 正运动学验证 |
| 牙模位置 | `(-34, -46, 463)mm`，`rpy (90, 0, 90)°` | 可达性 + 碰撞联合搜索 |

**注意：joint5 = 88.7°，正好在腕部奇异点上。** 这是该 TCP 姿态的固有性质，不是求解器问题——把目标平移 ±30mm，j5 只在 88.6~91.3 之间变化。真机从这个位姿启动时留意奇异报警。

生成结果：

```
waypoints       : 111
duration        : 39.9 s
IK failures     : 0
max joint step  : 1.4 deg
velocity usage  : 2%
validation      : PASS
tool-arm gap    : 77.4 mm (min at wp 64, Link4)
proximity gate  : PASS
```

## 2. 调整任务

### 改清洁区域

编辑 `config/tasks/arch_walk.yaml`：

```yaml
arch:
  u_range: [-0.030, 0.030]   # 只走牙弓中段（米，模型坐标系）
  lateral_offset_m: 0.003    # 向一侧偏 3 mm（刷颊侧/舌侧）
  trim_fraction: 0.05        # 两端各裁掉 5%
spacing_m: 0.005             # 相邻接触点间距
standoff_m: 0.002            # 刷头离牙面距离
max_tilt_deg: 40.0           # 工具相对牙合面的最大倾角

### 改工件摆位

`config/robot.yaml` 里的 `scene.model_to_base` 决定牙模在机器人坐标系里的位置。**这个值直接影响能不能避开碰撞**：

- 放在 `[0.45, 0, 0.25]`：机械臂折得太紧，200 mm 刷棒戳到上臂（实测 0.26 mm）
- 放在 `[0.65, 0, 0.30]`：同一条路径有 68 mm 间隙

改完一定要重跑 `plan`，看 `proximity gate` 是否 PASS。

### 换真机标定

dobot-pathgen calib --file 你的标定.yaml          # 先看解析出来的值对不对
dobot-pathgen calib --file 你的标定.yaml --apply  # 确认后写回 config/robot.yaml


## 3. Isaac Sim（可选，当前环境有问题）

### 状态：扩展全部加载成功，但启动后段错误

已经解决的问题：

| 问题 | 状态 |
|---|---|
| Git Bash 安装导致扩展符号链接是 Unix 路径 | 已修复（22 个链接改目录联接）|
| Kit 删除临时 zip 被 Windows 文件锁挡住 | 已绕过（逐个补链接）|
| `omni.isaac.ml_archive` 3.12 GB 下载不完整 | 已下载完成（断点续传）|

现在 Kit 能加载 **1169 行日志的全部扩展**，并打印 `app ready`，然后在 `SimulationApp()` 返回的瞬间于工作线程里发生 **access violation 段错误**。

已排除：
- 多 GPU（已强制 `multiGpu/enabled=False`）
- 线程数（已从 32 降到 8）
- 渲染器（试着关掉 RTX 也只推迟崩溃点）
- 扩展缺失（已确认全部加载）
- `extra_args` 格式（dict 会被静默忽略，已改成 list）

**最可能的原因**：机器上装了 **Todesk 虚拟显示适配器**（`Todesk Virtual Display Adapter`）。这类远程桌面虚拟显示驱动经常干扰 Vulkan 设备初始化。可以试着在设备管理器里禁用该适配器后重启，再跑一次。

这个崩溃发生在 Kit 内部，不在我们的代码里，我没有更多手段从外部绕过。

### 已写好的修复工具

```bat
python scripts\fix_isaac_links.py --dry-run    :: 只看会改什么
python scripts\fix_isaac_links.py              :: 修复扩展链接
python scripts\prefetch_isaac_ext.py --log out\isaac_unbuf.log --keep-zip
```

前一个用目录联接替换掉失效的符号链接；后一个用断点续传下载 Kit 拉不下来的大扩展包（ML 包 3.12 GB）。

### 建议做法

**用官方 Omniverse Launcher 装 Isaac Sim**，不要用 pip：

1. 下载 [NVIDIA Omniverse Launcher](https://www.nvidia.com/en-us/omniverse/download/)
2. 在 Launcher 里装 Isaac Sim 5.x
3. 装完后它自带 `python.bat`，路径类似 `C:\Users\<你>\AppData\Local\ov\pkg\isaac-sim-5.1.0\python.bat`
4. 然后：

```bat
cd D:\CompanyProject\DrobotGeneratPath
"C:\...\isaac-sim-5.1.0\python.bat" launch_isaac_sim.py --trajectory out\arch_lower_jaw.json
```

`launch_isaac_sim.py` 和 `src/dobot_pathgen/backends/isaac_sim.py` 已经写好了，只需换成官方安装的 Python。



## 4. 接真机（谨慎）

### 首次上电前

1. 控制柜切到 **TCP/IP 远程模式**
2. **关掉 DobotStudio Pro**（29999 端口只允许一个客户端）
3. 确认牙模和工具夹具锁紧
4. 手放在急停上

### 检查（不动机械臂）

python scripts/preflight.py --ip 192.168.5.1

会检查：轨迹合法性、关节限位、FK 复算、**工具-机械臂间隙**、网络连通、反馈流。

### 空载低速试跑

# 先只打印，不发送
python -m dobot_pathgen.cli execute --trajectory out/arch_lower_jaw.json --dry-run

# 再以 10% 速度空载运行
python -m dobot_pathgen.cli execute --trajectory out/arch_lower_jaw.json --ip 192.168.5.1 --speed-ratio 10

**首次务必空载**（刷子不接触牙模），确认运动方向和范围正确后，再降低 `standoff_m` 让它贴面。

### 走 ROS2 官方驱动

在 Ubuntu 22.04 + ROS2 Humble 里：

export IP_address=192.168.5.1
export DOBOT_TYPE=cr5
ros2 launch dobot_bringup_v4 dobot_bringup_ros2.launch.py    # 终端 1
ros2 launch dobot_moveit dobot_moveit.launch.py              # 终端 2

然后：

dobot-pathgen execute --trajectory out/arch_lower_jaw.json --backend ros2


## 5. 环境清单

| 用途 | 环境 | 说明 |
|---|---|---|
| 规划 / 校验 / 导查看器 | `D:\SoftWare\Anaconda\python.exe` (3.13) | 已装 `dobot-pathgen`（`pip install -e .`） |
| 离线视频渲染 | `D:\SoftWare\Anaconda\envs\Uclean` (3.8) | 需要 pyvista + imageio-ffmpeg |
| Isaac Sim | `D:\SoftWare\Anaconda\envs\isaacsim` (3.11) | 装了但启动失败，建议换官方安装 |
| ROS2 后端 | WSL Ubuntu 22.04 | 未配置 ROS2，需要自己装 Humble |

如果换机器或重装，基础依赖：

pip install numpy scipy PyYAML
pip install -e .
pip install pytest


## 6. 常见问题

**Q: 浏览器打开 viewer.html 是空白 / 一直"就绪"**
JSON 是异步加载的，用 `file://` 会被浏览器阻止。必须起本地 HTTP 服务（见第 1 节）。

**Q: 播放时机械臂是空心的 / 有破洞**
用了抽稀参数。重新生成时不要加 `--decimate`，或者写成 `--decimate 1`。

**Q: plan 返回退出码 3**
`proximity gate FAIL`：刷棒离机械臂太近。改 `model_to_base` 把工件挪远一点，或者换更短的刷头。

**Q: plan 返回退出码 4**
`IK dropped N poses`：有路点解不出来。通常是 `max_tilt_deg` 太大，或者区域跨过了牙弓弯曲处。减小倾角上限或缩小区域。

**Q: 轨迹能生成，但真机上运动方向不对**
检查 `model_to_base` 的 `rpy_deg`。牙模默认假设 Y 轴朝上、Z 轴朝前。你的扫描件如果朝向不同，改这里。

**Q: 刷头明明是 4 cm，为什么看起来只碰到一颗牙**
40 mm 直径的圆盘本来就比单颗牙（约 10 mm）宽 4 倍。贴面时圆盘边缘会压到相邻牙齿附近，这是几何事实。

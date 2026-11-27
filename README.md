# Auto_COC

面向《部落冲突》（Clash of Clans）19 本矿猪流（Miner/Hog + Dragon Duke）的自动化视觉感知与战术决策系统，基于 MuMu 模拟器 ADB 后台运行，无需占用窗口前台或鼠标。

## 核心能力

- **战前全图感知（YOLO26m + 零标注非防御建筑 + 主动探测）**
  - 双尺度（960/1280）YOLO26m 识别 24 类防御建筑、大本营与守卫（Guardian）
  - 战前自动探测喷火炮（Firespitter）150° 朝向扇区与 X 连弩（X-Bow）对地/对空模式
  - 零标注「草地剔除 + 红线连通块」自动检测外围非防御建筑，辅助清边与残局弓箭手收尾
- **44×44 空间势场进攻规划**
  - 综合大本营位置、巨石碑/守卫威胁、喷火炮正对惩罚、清边建筑密度自动决策龙王清边、女王冰人清边与大部队主攻线路
- **实时多目标追踪与法术/技能 FSM**
  - 纯 ADB 多点触控缩放 + `screenrecord` 低延迟实时视频流，支持 MuMu 后台静默运行
  - 龙王实时追踪与冰障（Ice Block）防集火保护、凤凰复活后自动判定复活法术（Revive）
  - 永王锚定大部队追踪、治疗（Healing）、冰冻（Freeze）、图腾（Totem）、毒药（Poison）与大守护者金身 / 女王 / 闰土技能自动释放
  - 全程自动生成带标注的复盘视频 `battle_annotated.mp4` 与决策日志 `run.json`

## 环境要求

1. **Python 3.10+**（已在 Python 3.13 + PyTorch 2.11 CUDA 12.8 验证，也支持 CPU 推理）
2. **MuMu 模拟器 12**
   - 分辨率必须设置为 **1800×1440**（5:4 画面比例，已标定坐标与 HUD 模板）
   - 开启模拟器设置中的 **ADB 调试**（默认端口通常为 `127.0.0.1:16384`）
3. **配兵栏位要求（16 格固定槽位）**
   - `1` 矿工（Miner）、`2` 野猪骑士（Hog Rider）、`3` 戈仑冰人（Ice Golem）、`4` 弓箭手（Archer）、`5` 德鲁伊（Druid）
   - `6` 训练营（Siege Barracks）
   - `7` 龙王（Dragon Duke，配凤凰战宠）、`8` 弓箭女皇（Archer Queen）、`9` 大守护者（Grand Warden）、`10` 飞盾战神（Royal Champion）
   - `11` 治疗法术、`12` 冰冻法术、`13` 图腾法术、`14` 冰障法术、`15` 毒药法术、`16` 复活法术

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/cloud107/Auto_COC.git
cd Auto_COC
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
```

如本机有 NVIDIA 显卡，建议先安装 CUDA 版 PyTorch（大幅加快战前双尺度 YOLO 推理）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

再安装其余依赖：

```bash
pip install -r requirements.txt
```

### 3. 配置 MuMu ADB 路径与端口

复制配置模板：

```bash
# Windows PowerShell:
Copy-Item config.example.json config.json
# macOS / Linux:
# cp config.example.json config.json
```

编辑 `config.json`，填入你的 MuMu `adb.exe` 路径和 ADB 端口：

```json
{
  "adb": "D:/MuMuPlayer/nx_main/adb.exe",
  "serial": "127.0.0.1:16384",
  "device": "auto",
  "output_dir": "runs"
}
```

- `device`: `"auto"` 自动优先使用 GPU (`0`)，无 CUDA 时自动回退到 `"cpu"`。
- 也可以不建 `config.json`，直接在命令行通过 `--adb` 和 `--serial` 传入。

### 4. 运行自动进攻

#### 方式 A：你已经手动进入侦察/开打界面（推荐，最通用）

在模拟器中打开要打的阵型（停在战前 30 秒倒计时画面，且尽量缩放到最远视野），然后运行：

```bash
python run_stage1_fsm.py --ready --full
```

#### 方式 B：私服/测试环境自动点击进入下一场并自动缩放到最远

```bash
python run_stage1_fsm.py --full
```

> 说明：程序全程通过 ADB 发送触摸事件与多点捏合缩放，**不需要**把 MuMu 保持在系统前台，也不会抢占你的鼠标。

### 5. 查看运行结果

每次运行会在 `runs/battle-YYYYMMDD-HHMMSS/`（或你指定的输出目录）下生成：

- `pre_yolo.jpg`：战前 YOLO 识别框、喷火炮朝向扇区、X 连弩模式、非防御建筑与进攻线路可视化
- `battle_annotated.mp4`：全过程 10fps 标注复盘视频（加 `--no-record-video` 可关闭录制）
- `run.json`：完整动作日志与最终摧毁率（Windows 下自动调用内置 WinRT OCR 识别结算界面百分比）

## 常用命令行参数

```bash
python run_stage1_fsm.py [reference_dir] [output_dir] [options]
```

| 参数 | 说明 |
| --- | --- |
| `--ready` | 当前已处于战前准备画面，跳过自动进战流程 |
| `--full` | 开启完整自动进攻（等价于 `--execute` + 全部法术/英雄技能/收尾开关） |
| `--execute` | 实际发送 ADB 操作（不加则仅预览规划不实际下兵） |
| `--adb <path>` | 覆盖 `config.json` 中的 ADB 路径 |
| `--serial <host:port>` | 覆盖 `config.json` 中的模拟器 ADB 序列号 |
| `--device <auto\|0\|cpu>` | 指定 YOLO 推理设备 |
| `--no-record-video` | 禁用 `battle_annotated.mp4` 视频录制以节省磁盘 |
| `--record-bc` | 额外保存 10Hz 行为克隆轨迹 `bc-trajectory.npz` |

## 仓库目录说明

```text
Auto_COC/
├── run_stage1_fsm.py          # 主入口：战前探测 + 开局部署 + 实时追踪与 FSM 控制
├── battle_grid_state.py       # 8×44×44 战场状态编码、喷火炮/连弩探测、非防御建筑识别、守卫追踪
├── potential_fsm.py           # 势场进攻规划与法术/英雄技能决策状态机
├── dragon_ice_block.py        # 龙王实时追踪与冰障法术防集火策略
├── healing_spell.py           # 永王锚定大部队追踪与治疗法术落点计算
├── battle_annotator.py        # 战前规划图与战斗标注视频渲染
├── battle_video_stream.py     # ADB screenrecord 低延迟实时视频流解码
├── recognize_hud.py           # 5:4 HUD 阶段识别与兵种/法术数量读取
├── capture_realistic_layouts.py # ADB 截图、触控与纯 ADB 后台双指缩放
├── th19-defense-ranges.json   # TH19 防御建筑射程与属性配置
├── read_windows_ocr.ps1       # Windows 结算界面摧毁率 OCR 脚本
├── config.example.json        # 本地配置模板
├── requirements.txt           # Python 依赖
├── models/
│   └── realistic-defense24-yolo26m-self-20261125.pt  # YOLO26m 建筑检测权重
└── captures/                  # HUD 参考模板与 44×44 棋盘投影标定文件
```

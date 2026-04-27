# Our_experiment/HCSAC 文件说明

这个文档用于整理 `Our_experiment/HCSAC` 文件夹下主要文件的功能、典型用法和输出位置。当前 HCSAC 目录大致分为 7 类：

- HCSAC 环境与 SAC 模型定义
- 模型保存/加载工具
- 轨迹与卸载可视化
- HCSAC 批量评估和热力图统计
- 风场种子分类与风场图生成
- 原始训练 notebook 和训练结果
- 旧版测试脚本与兼容文件

## 推荐使用流程

### 1. 确认模型和环境文件存在

HCSAC 的主要评估脚本会加载训练好的飞行动作模型和卸载动作模型：

```text
Our_experiment/HCSAC/data/sac_model_fly.pt
Our_experiment/HCSAC/data/sac_model_offload.pt
```

主环境文件：

```text
Our_experiment/HCSAC/ENV/UAVenv_SAC_Original.py
```

该环境中包含：

- UAV 搜索区域和网格参数
- 风场读取逻辑
- 地形/任务难度种子
- GBS/HAPS 基础设施种子
- UAV 飞行能耗和悬停功耗
- SAC 网络结构
- CNN 飞行动作策略和 GCN 卸载动作策略

### 2. 运行 HCSAC 三种风场指标统计

```bash
python Our_experiment/HCSAC/HCSAC_compare_metrics_by_wind.py \
  --infra-seed 999999 \
  --terrain-seed 10 \
  --wind-seeds 11,23,4800 \
  --traj-seed-mode random \
  --traj-seed-sample-size 10
```

默认约定：

- `11`：Low Wind
- `23`：Moderate Wind
- `4800`：Strong Wind

输出 JSON 默认保存到 HCSAC 的 `data` 目录，用于对比 GA 或论文表格。

### 3. 生成 HCSAC 卸载热力图

```bash
python Our_experiment/HCSAC/HCSAC_vis_offloading_seed_executor_range.py \
  --wind-seed 4800 \
  --terrain-seed 10 \
  --infra-seed 999999 \
  --traj-seed-mode random \
  --traj-seed-sample-size 10
```

该脚本会输出：

- BS/HAPS/LEO/CE 分设备卸载热力图
- terrain difficulty map
- wind field map
- markdown report
- 终端统计信息，包括 Average uncertainty 和 UAV lifetime 的 mean/std

### 4. 生成 HCSAC UAV 访问频率图

```bash
python Our_experiment/HCSAC/HCSAC_vis_offloading_visit_frequency_by_wind_class.py \
  --wind-seed 4800 \
  --terrain-seed 10 \
  --infra-seed 999999 \
  --traj-seed-mode random \
  --traj-seed-sample-size 10
```

该脚本会按 UAV 分别绘制访问频率热力图。

### 5. 如需风场分类，先生成风场 catalog

```bash
python Our_experiment/HCSAC/wind_seed_catalog_builder.py \
  --num-seeds 5000 \
  --subregion-size 20
```

输出包括：

- `wind_seed_catalog_5000.csv`
- `wind_seed_classes_5000.json`
- `wind_seed_classes_report_5000.md`

## 当前主线脚本

| 文件 | 功能 | 常用参数 | 输出 |
|---|---|---|---|
| `HCSAC_compare_metrics_by_wind.py` | 重新计算 HCSAC 在多组风场下的 Average uncertainty、coverage 和 UAV lifetime，并输出 JSON 和终端表格。 | `--infra-seed`、`--terrain-seed`、`--wind-seeds`、`--traj-seed-mode random/range`、`--traj-seed-sample-size` | `hcsac_metrics_by_wind_*.json` |
| `HCSAC_vis_offloading_seed_executor_range.py` | 当前推荐的 HCSAC 卸载热力图脚本。支持随机 10 个轨迹种子或连续 seed range，按卸载设备分别统计 BS/HAPS/LEO/CE。 | `--wind-seed`、`--terrain-seed`、`--infra-seed`、`--traj-seed-mode`、`--offload-metric count/frequency` | 卸载热力图、地形图、风场图、报告 md |
| `HCSAC_vis_offloading_visit_frequency_by_wind_class.py` | 当前推荐的 HCSAC UAV 访问频率图脚本。支持随机轨迹种子采样，按 UAV 分别绘制访问热力图。 | `--wind-seed`、`--terrain-seed`、`--infra-seed`、`--traj-seed-mode` | UAV visit frequency heatmap、报告 md |

## 环境和模型定义

| 文件 | 功能 | 说明 |
|---|---|---|
| `ENV/UAVenv_SAC_Original.py` | 当前主环境和 SAC 定义文件。包含 `UAVEnv` 和 `SAC`，是 GA/HCSAC 新脚本主要依赖的环境版本。 | 推荐作为当前实验主环境。 |
| `SAC_Original.py` | 早期 SAC 算法定义文件，包含 CNN policy/Q 网络、GCN policy/Q 网络、ReplayBuffer 和 SAC 训练逻辑。 | 主要用于早期 notebook 或训练实验参考。 |
| `UAVenv_Original.py` | 早期原始 UAV 环境。 | 主要用于旧版训练或对照，不建议作为当前新实验主入口。 |
| `UAVenv.py` | 早期/兼容 UAV 环境文件。 | 可能被 `SAC_Original.py` 或旧 notebook 使用。 |
| `ENV/dist/UAVenv_SAC.py` | PyArmor 加密/打包后的旧版环境。 | 旧版测试脚本依赖。当前建议优先使用 `ENV/UAVenv_SAC_Original.py`。 |
| `ENV/dist/pyarmor_runtime_000000/` | PyArmor 运行时文件。 | 用于运行 `ENV/dist/UAVenv_SAC.py`。 |

## 模型保存和加载

| 文件 | 功能 | 说明 |
|---|---|---|
| `UAV_SAVE.py` | 保存和加载 SAC agent 权重，以及保存/读取训练历史。 | 主要函数：`save_sac_agent`、`load_sac_agent`、`save_training_history`、`load_training_history`。 |

常用模型路径：

```text
Our_experiment/HCSAC/data/sac_model_fly.pt
Our_experiment/HCSAC/data/sac_model_offload.pt
```

## 轨迹和卸载可视化工具

| 文件 | 功能 | 说明 |
|---|---|---|
| `UAV_VIS_offloading_2.py` | 当前主要的 HCSAC 轨迹/卸载执行与可视化函数。很多批量统计脚本会 import 其中的 `visualize_trajectory`。支持 wind seed、terrain seed、traj seed、infra seed 分离。 | 当前推荐使用。 |
| `UAV_VIS_offloading.py` | 较早版本的 offloading 可视化工具。 | 旧脚本兼容。 |
| `UAV_VIS_without_offloading.py` | 无卸载版本的轨迹可视化。 | 用于 no-offloading 对照。 |
| `UAV_VIS.py` | 更早的基础轨迹可视化。 | 旧版展示脚本使用。 |

## 单 seed 和 seed range 测试脚本

| 文件 | 功能 | 说明 |
|---|---|---|
| `vis_offloading_seed_executor.py` | 单个 seed 执行器。支持分别指定 `wind-seed`、`terrain-seed`、`traj-seed`、`infra-seed`，并可选择是否展示。 | 适合复现某一个 seed 的轨迹和卸载热力图。 |
| `vis_offloading_seed_search.py` | 在给定 seed 范围内搜索 Average uncertainty 最低的轨迹种子，并输出对应热力图。 | 早期用于找最佳 seed。 |
| `vis_offloading_seed_executor_range.py` | 旧版范围统计脚本。对一个 wind seed 下的一段轨迹 seed 进行卸载热力图聚合。 | 当前更推荐 `HCSAC_vis_offloading_seed_executor_range.py`。 |
| `vis_offloading_visit_frequency_by_wind_class.py` | 旧版 UAV 访问频率图脚本，统计一个风场下多个 traj seed 的访问频率。 | 当前更推荐 `HCSAC_vis_offloading_visit_frequency_by_wind_class.py`。 |

典型单 seed 命令：

```bash
python Our_experiment/HCSAC/vis_offloading_seed_executor.py \
  --wind-seed 4800 \
  --terrain-seed 10 \
  --traj-seed 90 \
  --infra-seed 999999 \
  --show
```

## 风场工具脚本

| 文件 | 功能 | 常用输出 |
|---|---|---|
| `wind.py` | 风场基础工具函数。包括从 `wind.json` 提取子区域、计算风速均值、风向标准差等。 | 被环境和 catalog 脚本 import。 |
| `wind_seed_catalog_builder.py` | 遍历风场 seed，统计每个 seed 的平均风速，并按三分位划分 Low/Moderate/Strong Wind。 | CSV、JSON、Markdown 报告 |
| `vis_wind_field_from_catalog.py` | 从 `wind_seed_classes_5000.json` 读取指定风场 seed，并画风场图。支持大字体、稀疏加粗箭头。 | 单张风场 PNG |
| `vis_wind_seed_direction_contrast.py` | 寻找与参考 wind seed 平均风速相近、但风向差异较大的 top-k wind seeds，并输出对比图和报告。 | combined PNG、individual PNG、JSON、md |

典型命令：

```bash
python Our_experiment/HCSAC/vis_wind_field_from_catalog.py \
  --wind-seed 4800 \
  --output-path Our_experiment/HCSAC/data/wind_field_w4800_large.png \
  --arrow-step 2 \
  --arrow-width 0.006
```

```bash
python Our_experiment/HCSAC/vis_wind_seed_direction_contrast.py \
  --reference-wind-seed 4800 \
  --candidate-seed-start 0 \
  --candidate-seed-end 4999 \
  --top-k 5
```

## 原始训练 notebook

| 文件 | 功能 |
|---|---|
| `HC_SAC_Original.ipynb` | SAC/HCSAC 原始训练 notebook。 |
| `HC_DQN_Original.ipynb` | DQN 对比算法训练/实验 notebook。 |
| `HC_PPO_Original.ipynb` | PPO 对比算法训练/实验 notebook。 |
| `HC_TRPO_Original.ipynb` | TRPO 对比算法训练/实验 notebook。 |
| `HC_A2C_Original.ipynb` | A2C 对比算法训练/实验 notebook。 |
| `data/Comparison_of_Experimental_Algorithms.ipynb` | 训练曲线或算法对比展示 notebook。 |

## 早期测试脚本

| 文件 | 功能 | 说明 |
|---|---|---|
| `vis_offloading_testing.py` | 早期 offloading 展示脚本。 | 依赖旧版 dist 环境。 |
| `vis_offloading_testing2.py` | 固定 seed 的 offloading 展示，并额外输出总卸载热力图。 | 后续已被更细分设备热力图脚本替代。 |
| `vis_offloading_testing3.py` | 固定 seed 的 offloading 展示，并按 BS/HAPS/LEO/CE 分设备输出热力图。 | 早期展示用。 |
| `vis_without_offloading_testing.py` | 无卸载轨迹展示脚本。 | 旧版 dist 环境。 |

## 数据目录说明

| 文件/目录 | 功能 |
|---|---|
| `data/sac_model_fly.pt` | 训练好的飞行动作 SAC 模型。 |
| `data/sac_model_offload.pt` | 训练好的卸载动作 SAC/GCN 模型。 |
| `data/*returns_random.pkl` | 不同算法的训练 return 数据，如 SAC、DQN、PPO、TRPO、A2C。 |
| `data/wind_data.pkl` | 风场实验数据。 |
| `data/wind_*_effects.png` | 风速/风向相关实验图。 |
| `data/wind_direction_contrast/` | 风向差异对比实验输出。 |
| `Our_experiment/HCSAC/Our_experiment/HCSAC/data/` | 历史脚本因相对路径生成的嵌套输出目录。不是主要源码目录。 |
| `wind.json` | HCSAC 目录下的风场原始 JSON 数据。 |
| `ENV/dist/wind.json` | dist 环境使用的风场 JSON。 |

## 常见指标

- `Average uncertainty`：平均不确定度，越低越好。
- `Coverage`：覆盖率，通常计算为 `coverage = 1 - average_uncertainty`，表格中可能显示为百分比。
- `UAV lifetime`：无人机生命周期，通常会输出 mean/std。
- `Offloading Count`：卸载次数，更适合说明某区域对某个设备的选择频次。
- `Offloading Frequency`：卸载频率，适合不同 seed 数量或不同总步数下归一化比较。
- `Visit Frequency`：UAV 访问频率，通常按 UAV 分图展示。

## seed 说明

当前推荐把随机性拆成多个 seed：

- `wind-seed`：控制风场子区域。
- `terrain-seed`：控制地形/任务难度矩阵。
- `traj-seed`：控制轨迹执行过程中的随机性。
- `infra-seed`：控制 GBS/HAPS 等基础设施位置。

旧脚本中的 `--seed` 往往会同时影响多个随机源；如果要和 GA 或新 HCSAC 结果严格对应，应优先使用新脚本中分离后的 seed 参数。

## 建议使用优先级

新实验优先使用：

1. `ENV/UAVenv_SAC_Original.py`
2. `UAV_VIS_offloading_2.py`
3. `HCSAC_compare_metrics_by_wind.py`
4. `HCSAC_vis_offloading_seed_executor_range.py`
5. `HCSAC_vis_offloading_visit_frequency_by_wind_class.py`
6. `wind_seed_catalog_builder.py`
7. `vis_wind_field_from_catalog.py`
8. `vis_wind_seed_direction_contrast.py`

旧版或参考用：

- `vis_offloading_seed_executor_range.py`
- `vis_offloading_visit_frequency_by_wind_class.py`
- `vis_offloading_testing.py`
- `vis_offloading_testing2.py`
- `vis_offloading_testing3.py`
- `vis_without_offloading_testing.py`
- `ENV/dist/UAVenv_SAC.py`

## 注意事项

- 批量统计脚本默认会设置 `SDL_VIDEODRIVER=dummy` 和 `SDL_AUDIODRIVER=dummy`，用于无界面批处理。
- 如果需要真实弹窗展示轨迹，使用单 seed executor 或 testing 脚本，并确保当前环境支持 Pygame 窗口。
- 对比和 heatmap 脚本会加载 PyTorch 模型，运行前需要确认模型文件存在。
- `HCSAC_vis_*` 新脚本默认更适合和 GA 随机 10 个轨迹种子的评估方式对齐。
- 如果输出路径出现嵌套的 `Our_experiment/HCSAC/Our_experiment/HCSAC/data`，通常是历史相对路径造成的输出位置问题，不影响当前主线脚本逻辑。

# Our_experiment/GA 文件说明

这个文档用于整理 `Our_experiment/GA` 文件夹下主要 Python 文件的功能、典型用法和输出位置。当前 GA 实验代码大致分为 6 类：

- GA 部署搜索：寻找 UAV 初始部署点和目标点。
- GA 结果回放画图：读取 GA 搜索结果，再生成卸载热力图、访问频率图、地形图、风场图。
- 算法对比实验：比较 offloading / no-offloading / no-GA / rule-based 等实验组。
- UAV 数量对比实验：比较不同 UAV 数量下覆盖率变化。
- 画图重绘工具：读取已有 JSON，重新生成更适合论文展示的图。
- 早期测试和工具文件：旧版 demo、pickle 工具和公共函数。

## 推荐运行流程

### 1. 先运行 GA 搜索，得到最优部署点

完整 GA + offloading 版本：

```bash
python Our_experiment/GA/ga_deployment_seed_search_2.py \
  --infra-seed 999999 \
  --terrain-seed 10 \
  --wind-seed 4800 \
  --traj-seed-min 0 \
  --traj-seed-max 200 \
  --traj-seed-sample-size 10 \
  --iterations 20 \
  --population-size 12
```

无卸载版本：

```bash
python Our_experiment/GA/ga_deployment_seed_search_2_no_offloading.py \
  --infra-seed 999999 \
  --terrain-seed 10 \
  --wind-seed 4800
```

简单 greedy 轨迹 + offloading 版本：

```bash
python Our_experiment/GA/ga_deployment_seed_search_2_rule_based_offloading_2.py \
  --infra-seed 999999 \
  --terrain-seed 10 \
  --wind-seed 4800
```

常用种子含义：

- `--infra-seed`：控制 GBS/HAPS 等基础设施位置。
- `--terrain-seed`：控制地形/任务难度矩阵。
- `--wind-seed`：控制风场。
- `--traj-seed-*`：控制轨迹评估中的随机轨迹种子采样池。

### 2. 再读取 GA 结果生成热力图

推荐用合并版，避免重复 rollout：

```bash
python Our_experiment/GA/GA_vis_offloading_visit_combined.py \
  --ga-result-json Our_experiment/GA/data/ga_best_deployment_random10_w4800_g10_i999999_pool0_200.json \
  --output-dir Our_experiment/GA/data
```

该脚本会同时输出：

- offloading heatmap
- UAV visit frequency heatmap
- terrain difficulty map
- wind field map
- markdown report
- summary JSON

默认会在 `output-dir` 下按种子创建子文件夹，例如：

```text
Our_experiment/GA/data/wind_4800_terrain_10_infra_999999/
```

### 3. 做三种风场下算法对比

```bash
python Our_experiment/GA/compare_ga_algorithms_by_wind.py \
  --infra-seed 999999 \
  --terrain-seed 10 \
  --wind-seeds 11,23,4800
```

其中默认约定：

- `11`：Low Wind
- `23`：Moderate Wind
- `4800`：Strong Wind

如果还需要 `No-GA + No-Offloading` 补充组：

```bash
python Our_experiment/GA/compare_ga_algorithms_by_wind_supplement.py \
  --infra-seed 999999 \
  --terrain-seed 10 \
  --wind-seeds 11,23,4800
```

然后打印论文风格表格：

```bash
python Our_experiment/GA/print_ga_algorithm_comparison_table.py
```

## GA 搜索脚本

| 文件 | 功能 | 典型输入 | 主要输出 |
|---|---|---|---|
| `ga_deployment_seed_search_2.py` | 当前主要的 GA + offloading 部署搜索脚本。轨迹动作使用训练好的 SAC 飞行动作策略，卸载动作使用 SAC/GCN 卸载策略。每次 GA iteration 从轨迹种子池中随机抽取若干种子评估。 | `--infra-seed`、`--terrain-seed`、`--wind-seed`、`--traj-seed-min/max`、`--traj-seed-sample-size`、`--iterations`、`--population-size` | `Our_experiment/GA/data/ga_best_deployment_random{sample_size}_w{wind_seed}_g{terrain_seed}_i{infra_seed}_pool{seed_min}_{seed_max}.json` 和 `.pkl` |
| `ga_deployment_seed_search_2_no_offloading.py` | GA 部署搜索，但关闭卸载，所有任务固定本地处理。用于比较 offloading 带来的收益。 | 与 `ga_deployment_seed_search_2.py` 基本一致 | `ga_best_deployment_nooffload_random...json` 和 `.pkl` |
| `ga_deployment_seed_search_2_rule_based_offloading_2.py` | GA 部署搜索，轨迹动作使用简单 greedy rule-based 策略，卸载动作仍使用 offload agent。用于比较 RL 轨迹策略与简单规则策略。 | 与 `ga_deployment_seed_search_2.py` 基本一致 | simple-greedy offloading 对应的 best deployment JSON/PKL |
| `ga_deployment_seed_search_2_rule_based_offloading.py` | 较早的 rule-based trajectory + offloading 版本，规则中使用更多环境信息。现在一般优先使用 `_2.py` 简单 greedy 版本做公平对照。 | 与 `ga_deployment_seed_search_2.py` 基本一致 | rule-based offloading 对应的 best deployment JSON/PKL |
| `ga_deployment_seed_search.py` | 旧版 GA 搜索脚本，和 `GA_Original.ipynb` 更接近。使用固定轨迹种子范围 `traj-seed-start/end`，而不是每轮随机抽样。 | `--traj-seed-start`、`--traj-seed-end`、GA 参数 | `ga_best_deployment_w{wind_seed}_g{terrain_seed}_i{infra_seed}_t{start}_{end}.json` 和 `.pkl` |
| `ga_deployment_seed_search_2_coverage_convergence.py` | 专门生成 GA 收敛曲线。基于 `ga_deployment_seed_search_2.py` 的 GA 逻辑，记录每代 coverage max/mean/min。coverage 计算为 `1 - uncertainty`。 | GA 参数、种子参数、`--plot-path` | GA coverage convergence PNG |

## GA 结果回放和热力图脚本

| 文件 | 功能 | 典型用法 | 输出 |
|---|---|---|---|
| `GA_vis_offloading_visit_combined.py` | 推荐使用的合并版回放脚本。读取 GA best deployment JSON，一次 rollout 同时统计卸载热力图和 UAV 访问频率图，避免重复计算。 | `python Our_experiment/GA/GA_vis_offloading_visit_combined.py --ga-result-json <json> --output-dir Our_experiment/GA/data` | offloading heatmap、visit heatmap、terrain map、wind map、reports、summary JSON |
| `GA_vis_offloading_seed_executor_range.py` | 新版 GA 卸载热力图脚本。读取 `ga_deployment_seed_search_2.py` 的随机轨迹种子格式结果，统计 BS/HAPS/LEO/CE 等设备的卸载次数或频率。 | `--ga-result-json`、`--seed-source`、`--offload-metric count/frequency` | 卸载热力图、地形图、风场图、报告 md |
| `GA_vis_offloading_visit_frequency_by_wind_class.py` | 新版 GA UAV 访问频率图脚本。读取 `ga_deployment_seed_search_2.py` 的结果，按 UAV 分别画访问频率热力图。默认支持随机 10 个轨迹种子。 | `--ga-result-json`、`--seed-source random-sample` | UAV visit frequency heatmap、报告 md |
| `vis_offloading_seed_executor_range.py` | 旧版 GA 卸载热力图脚本，主要配合 `ga_deployment_seed_search.py` 的固定 seed range 结果。 | `--ga-result-json`、`--traj-seed-start/end` | 卸载热力图、地形图、风场图、报告 md |
| `vis_offloading_visit_frequency_by_wind_class.py` | 旧版 GA UAV 访问频率脚本，主要配合 `ga_deployment_seed_search.py` 的固定 seed range 结果。 | `--ga-result-json`、`--traj-seed-start/end` | UAV visit frequency heatmap、报告 md |

### `seed-source` 说明

新版 `GA_vis_*` 脚本支持以下轨迹种子来源：

- `auto`：自动选择，通常会根据结果 JSON 格式选择合适模式。
- `random-sample`：从轨迹种子池随机抽取，默认常用于随机 10 个种子的评估。
- `best-iteration`：使用 GA 最优 iteration 记录的轨迹种子集合。
- `all-iterations`：合并 GA 每轮采样过的轨迹种子。
- `range`：使用 `--traj-seed-start` 到 `--traj-seed-end` 的连续范围。

## 算法对比实验脚本

| 文件 | 功能 | 默认比较对象 | 输出 |
|---|---|---|---|
| `compare_ga_algorithms_by_wind.py` | 从零重新运行并比较不同算法在三种风场下的 lifetime 和 coverage。 | `GA + Offloading`、`GA + No-Offloading`、`Simple Greedy + Offloading`、`No-GA + Offloading` | `Our_experiment/GA/data/ga_algorithm_comparison_i999999_g10_winds11_23_4800.json` |
| `compare_ga_algorithms_by_wind_supplement.py` | 补充统计 `No-GA + No-Offloading`，并单独保存 JSON。 | `No-GA + No-Offloading` | `Our_experiment/GA/data/ga_algorithm_comparison_supplement_i999999_g10_winds11_23_4800.json` |
| `print_ga_algorithm_comparison_table.py` | 读取主对比 JSON 和 supplement JSON，在终端打印表格。表格显示 lifetime 和 coverage 的 mean/std。 | 读取已有 JSON，不重新计算 | Markdown 或 plain terminal table |

常用命令：

```bash
python Our_experiment/GA/compare_ga_algorithms_by_wind.py \
  --infra-seed 999999 \
  --terrain-seed 10 \
  --wind-seeds 11,23,4800

python Our_experiment/GA/compare_ga_algorithms_by_wind_supplement.py \
  --infra-seed 999999 \
  --terrain-seed 10 \
  --wind-seeds 11,23,4800

python Our_experiment/GA/print_ga_algorithm_comparison_table.py --format markdown
```

## UAV 数量对比脚本

| 文件 | 功能 | 输出 |
|---|---|---|
| `compare_uav_count_three_groups.py` | 比较不同 UAV 数量下三组实验的最终覆盖率和不确定度：`GA + Offloading`、`GA + No-Offloading`、`No-GA + Offloading`。默认 UAV 数量为 `1,2,3,4,5,6`。 | JSON、coverage plot、uncertainty plot |
| `replot_uav_count_coverage_large_font.py` | 读取 UAV 数量对比 JSON，重新绘制大字体 coverage 折线图。适合论文或汇报。 | 大字体 PNG |
| `replot_uav_count_coverage_mean_std.py` | 读取 UAV 数量对比 JSON，用 mean 画折线，并用 std 画阴影带。标题不带种子。 | mean/std coverage PNG |

典型命令：

```bash
python Our_experiment/GA/compare_uav_count_three_groups.py \
  --infra-seed 999999 \
  --terrain-seed 10 \
  --wind-seed 4800 \
  --uav-counts 1,2,3,4,5,6

python Our_experiment/GA/replot_uav_count_coverage_mean_std.py \
  --input-json Our_experiment/GA/data/mul_uav/ga_three_group_uav_count_comparison_w4800_g10_i999999.json
```

## 公共工具和早期测试文件

| 文件 | 功能 | 说明 |
|---|---|---|
| `ga_vis_common.py` | GA 可视化和评估公共工具。包括路径处理、JSON 读写、seed 设置、环境和 SAC agent 加载、deployment position 校验、单次 rollout 等。 | 被大部分 GA 脚本 import，不建议单独运行。 |
| `PickleTA.py` | 简单 pickle 保存/读取工具，用于保存 `best_solution`、`best_fitness`、轨迹动作和卸载动作。 | 早期实验工具。 |
| `GA_Optimal_population_testing_0.py` | 早期测试脚本。读取 `results.pkl` 中的 GA 最优结果，再调用旧版可视化展示轨迹和卸载。 | 依赖 `Our_experiment/HCSAC/ENV/dist` 和旧版 `UAV_VIS_offloading`。 |
| `GA_without_Optimal_population_testing_0.py` | 早期测试脚本。不读取 GA 最优部署，直接调用旧版可视化进行对照展示。 | 依赖旧版 dist 环境。 |
| `__init__.py` | Python package 标记文件。 | 不需要手动运行。 |

## Notebook 和数据文件

| 文件/目录 | 功能 |
|---|---|
| `GA_Original.ipynb` | 原始 GA notebook。部分后续 `.py` 文件是根据该 notebook 逻辑改写而来。 |
| `results.pkl` | 早期 pickle 实验结果，主要给 `GA_Optimal_population_testing_0.py` 使用。 |
| `wind.json` | GA 文件夹内的风场相关数据文件。 |
| `data/` | 当前主要实验输出目录，保存 JSON、PKL、PNG 和报告 md。 |
| `Our_experiment/GA/data/` 的嵌套副本 | 如果出现 `Our_experiment/GA/Our_experiment/GA/data/...` 这种路径，通常是旧脚本使用相对路径时生成的历史输出，不是主要源码。 |

## 常见输出指标

- `Average uncertainty`：平均不确定度，越低越好。
- `Coverage`：覆盖率，通常按 `coverage = 1 - average_uncertainty` 计算；部分表格中显示为百分比。
- `Lifetime`：UAV 平均生命周期，通常终端输出 mean/std。
- `Offloading Count`：卸载次数，更适合解释“哪些区域更频繁选择某个卸载设备”。
- `Offloading Frequency`：卸载频率，适合跨不同 seed 数量或总步数归一化比较。

## 建议使用优先级

新实验优先使用：

1. `ga_deployment_seed_search_2.py`
2. `ga_deployment_seed_search_2_no_offloading.py`
3. `ga_deployment_seed_search_2_rule_based_offloading_2.py`
4. `GA_vis_offloading_visit_combined.py`
5. `compare_ga_algorithms_by_wind.py`
6. `compare_ga_algorithms_by_wind_supplement.py`
7. `print_ga_algorithm_comparison_table.py`
8. `compare_uav_count_three_groups.py`
9. `replot_uav_count_coverage_mean_std.py`

旧版或仅作兼容/参考：

- `ga_deployment_seed_search.py`
- `vis_offloading_seed_executor_range.py`
- `vis_offloading_visit_frequency_by_wind_class.py`
- `GA_Optimal_population_testing_0.py`
- `GA_without_Optimal_population_testing_0.py`

## 注意事项

- 多数脚本会加载 HCSAC 训练好的飞行和卸载模型，因此运行前需要确认 `Our_experiment/HCSAC/data/sac_model_fly` 和 `Our_experiment/HCSAC/data/sac_model_offload` 存在。
- 对比脚本会从零重新运行 GA 和 rollout，运行时间可能比较长。
- 画图脚本如果只是读取已有 JSON 或 GA result JSON，通常比完整 GA 搜索快很多。
- 如果需要论文图，优先使用 `replot_*` 脚本或 `GA_vis_offloading_visit_combined.py` 的输出，再按需要调字体参数。

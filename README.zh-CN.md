<p align="center">
  <img src="./assets/Fail2Drive_logo.png" width="400"/>
</p>

<h1 align="center">Fail2Drive：闭环驾驶泛化能力基准测试</h1>

<p align="center"><a href="./README.md">English</a> · <a href="./README.zh-CN.md">简体中文</a> · <a href="./README.ja.md">日本語</a></p>

<p align="center">
  <a href="https://simonger.github.io/fail2drive/">项目主页</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/pdf/2604.08535">论文</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/datasets/SimonGer/Fail2Drive">下载</a> &nbsp;|&nbsp;
  <a href="https://discord.gg/HZ83Em6kyZ">Discord</a> &nbsp;|&nbsp;
  <a href="https://github.com/SimonGer/fail2drive_scenario_hub">场景中心</a>
</p>

<p align="center">
  <img src="./assets/hero.gif" alt="预览图" width="61.3%"> <img src="./assets/bar_plot.png" width="37.7%">
</p>

Fail2Drive 是首个专为测试真正未见长尾场景中闭环泛化能力而设计的 CARLA v2 基准。通过为每条分布偏移路线配对一个分布内参考场景，它揭示了当前最先进驾驶模型中大量隐藏的失效模式。

## 亮点

- 17 个未见场景，用于评估真实泛化能力
- 30 种新资源，包括动物、视觉噪声和对抗性障碍物
- 配对路线设计支持量化泛化差距
- 100 对路线，涵盖多样化环境和配置
- 用于创建自定义障碍物和路线的工具箱

## 排行榜

[![排行榜](https://raw.githubusercontent.com/SimonGer/fail2drive_leaderboard/main/rendered/table.png)](https://github.com/SimonGer/fail2drive_leaderboard)

## 目录

- [安装](#安装)
- [实验](#实验)
- [评估](#评估)
- [Fail2Drive 工具箱](#fail2drive-工具箱)

## 安装

> 如果你希望将 Fail2Drive 引入现有 CARLA 项目，我们在 [plugin 分支](https://github.com/autonomousvision/Fail2Drive/tree/plugin)提供了轻量级插件安装方式。以下安装包含 `carla_garage` 模型，可作为新用户的起点。

完成快速安装后，你就可以手动探索该基准、运行基线智能体，并开始测试自定义场景。

```bash
# 1. Clone this repository
git clone https://github.com/autonomousvision/fail2drive.git
cd fail2drive

# 2. Set up the Fail2Drive CARLA simulator
mkdir f2d_carla
curl -L \
  https://huggingface.co/datasets/SimonGer/fail2drive/resolve/main/fail2drive_simulator.tar.gz \
  | tar -xz -C f2d_carla

# 3. Create the conda environment
conda env create -f environment.yml
conda activate fail2drive

# NOTE: The pip installed carla==0.9.15 should work, but may cause warnings in some places.
# If you want to install the official Fail2Drive PythonAPI you can find it at:
# f2d_carla/PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl

# 4. Set environment variables
source env_vars.sh

# Ready to start experimenting!
```

## 实验

要运行以下任一实验，请在第二个终端中启动 CARLA：

```bash
bash ${CARLA_ROOT}/CarlaUE4.sh
```

<details>
<summary>计算资源有限时的使用技巧</summary>

- 使用 `-RenderOffscreen` 运行 CARLA，以避免打开观察者窗口。
- 使用 `-quality-level=Low` 运行 CARLA，以降低渲染成本。最终评估时请勿使用此选项。
- 通过指定 `-graphicsadapter=[id]`，让 CARLA 和模型在不同 GPU 上运行。

</details>

#### 你能手动完成基准路线吗？

```bash
python leaderboard/leaderboard/leaderboard_evaluator.py \
  --agent ${WORK_DIR}/leaderboard/leaderboard/autoagents/human_agent_keyboard.py \
  --routes ${WORK_DIR}/fail2drive_split/Generalization_PedestriansOnRoad_1085.xml
```

#### 运行 PDM-Lite 专家策略

```bash
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent ${WORK_DIR}/team_code/visu_agent.py \
  --track MAP \
  --routes ${WORK_DIR}/fail2drive_split/Generalization_PedestriansOnRoad_1085.xml
```

#### 运行 TransFuser++ 模型

运行模型前，请将检查点下载到 `checkpoints` 文件夹：

```bash
mkdir -p checkpoints/tfpp
wget -P checkpoints/tfpp \
  https://huggingface.co/SimonGer/TFv5/resolve/main/all_towns/model_0030_0.pth \
  https://huggingface.co/SimonGer/TFv5/resolve/main/all_towns/config.json
```

然后使用 `LIVE_VISU` 标志运行 TransFuser++，以实时查看模型输入：

```bash
LIVE_VISU=1 python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --routes ${WORK_DIR}/fail2drive_split/Generalization_PedestriansOnRoad_1085.xml \
  --agent ${WORK_DIR}/team_code/sensor_agent.py \
  --agent-config ${WORK_DIR}/checkpoints/tfpp
```

## 评估

### Fail2Drive 规则

作为离线基准，用户可以访问完整的评估路线和资源。为了防止过拟合并维持 Fail2Drive 的公平性，我们制定了以下规则：

1. **不得使用 Fail2Drive 场景进行训练。** 模型不得使用 Fail2Drive 引入的路线、场景定义或资源进行训练或微调。该基准严格作为留出测试集使用。

2. **允许外部预训练。** 可以在大规模现实世界数据集、互联网规模的多模态语料库、基础模型或 VLM/LLM 主干网络上进行预训练。这类通用视觉或语言知识被视为模型先验，不构成对基准规则的违反。

3. **提交排行榜成绩。** 我们鼓励用户通过 Pull Request 向[官方排行榜仓库](https://github.com/SimonGer/fail2drive_leaderboard)提交最终分数。这样可以保持比较一致，并促进透明的基准评测。更多信息请参阅[贡献指南](https://github.com/SimonGer/fail2drive_leaderboard?tab=contributing-ov-file)。

### SLURM 评估

我们提供了使用 SLURM 集群在完整基准上评估模型的工具。[slurm_evaluate.py](slurm_evaluate.py) 脚本会自动提交作业，提交数量上限由 [eval_num_jobs.txt](eval_num_jobs.txt) 指定，并持续监控评估过程。你需要进行少量修改，使该脚本适配具体集群和模型。请在脚本中查找 `NOTE` 和 `TODO` 注释。

> **注意：** 运行脚本前，请务必激活 conda 环境：`conda activate fail2drive`

如果评估期间遇到问题，欢迎提交 Issue！

### 结果生成

要获得最终分数，请使用 [tools/f2d_result_parser.py](tools/f2d_result_parser.py) 脚本：

```bash
python tools/f2d_result_parser.py /path/to/results --method MyMethod
```

## Fail2Drive 工具箱

我们提供了多种工具，可利用 Fail2Drive 的可自定义场景和新资源生成新路线。

![工具箱](./assets/toolbox.png)

你可以在[此处](toolbox)查看这些工具的文档。若要分享和浏览社区创建的场景，请访问我们的[场景中心](https://github.com/SimonGer/fail2drive_scenario_hub)。

## 致谢

本项目建立在众多优秀开源项目的成果之上。我们大量使用了 [carla_garage](https://github.com/autonomousvision/carla_garage) 和 [carla_route_generator](https://github.com/autonomousvision/carla_route_generator)，并集成了 [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard) 与 [scenario_runner](https://github.com/carla-simulator/scenario_runner) 框架。

我们还要感谢本基准所评估模型的作者：[SimLingo](https://github.com/RenzKa/simlingo)、[HiP-AD](https://github.com/nullmax-vision/HiP-AD)、[Orion](https://github.com/xiaomi-mlab/Orion)、[PlanT2](https://github.com/autonomousvision/plant2) 和 [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo)。

## 引用

```bibtex
@misc{gerstenecker2026fail2drivebenchmarkingclosedloopdriving,
      title={Fail2Drive: Benchmarking Closed-Loop Driving Generalization},
      author={Simon Gerstenecker and Andreas Geiger and Katrin Renz},
      year={2026},
      eprint={2604.08535},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2604.08535},
}
```

<p align="center">
  <img src="./assets/Fail2Drive_logo.png" width="400"/>
</p>

<h1 align="center">Fail2Drive：クローズドループ運転の汎化性能ベンチマーク</h1>

<p align="center"><a href="./README.md">English</a> · <a href="./README.zh-CN.md">简体中文</a> · <a href="./README.ja.md">日本語</a></p>

<p align="center">
  <a href="https://simonger.github.io/fail2drive/">プロジェクトページ</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/pdf/2604.08535">論文</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/datasets/SimonGer/Fail2Drive">ダウンロード</a> &nbsp;|&nbsp;
  <a href="https://discord.gg/HZ83Em6kyZ">Discord</a> &nbsp;|&nbsp;
  <a href="https://github.com/SimonGer/fail2drive_scenario_hub">シナリオハブ</a>
</p>

<p align="center">
  <img src="./assets/hero.gif" alt="プレビュー" width="61.3%"> <img src="./assets/bar_plot.png" width="37.7%">
</p>

Fail2Drive は、真に未知のロングテールシナリオに対するクローズドループの汎化性能を検証するために設計された、初の CARLA v2 ベンチマークです。分布がシフトした各ルートを分布内の参照シナリオと組み合わせることで、現在の最先端運転モデルに潜む重大な失敗モードを明らかにします。

## 特長

- 真の汎化性能を評価するための 17 個の未知シナリオ
- 動物、視覚ノイズ、敵対的障害物を含む 30 種類の新規アセット
- ペア化されたルート設計により汎化ギャップを定量化可能
- 多様な環境と設定にわたる 100 組のルート
- カスタム障害物とルートを作成するためのツールボックス

## リーダーボード

[![リーダーボード](https://raw.githubusercontent.com/SimonGer/fail2drive_leaderboard/main/rendered/table.png)](https://github.com/SimonGer/fail2drive_leaderboard)

## 目次

- [インストール](#インストール)
- [実験](#実験)
- [評価](#評価)
- [Fail2Drive ツールボックス](#fail2drive-ツールボックス)

## インストール

> 既存の CARLA プロジェクトに Fail2Drive を導入したい場合は、[plugin ブランチ](https://github.com/autonomousvision/Fail2Drive/tree/plugin)で軽量なプラグイン形式のインストール方法を提供しています。以下のインストールには、新規ユーザーの出発点となる `carla_garage` モデルが含まれています。

簡単なインストールが完了すれば、ベンチマークを手動で探索し、ベースラインエージェントを実行して、カスタムシナリオのテストを開始できます。

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

## 実験

以下の実験を実行するには、2 つ目のターミナルで CARLA を起動します。

```bash
bash ${CARLA_ROOT}/CarlaUE4.sh
```

<details>
<summary>計算リソースが限られている場合のヒント</summary>

- 観戦用ウィンドウが開かないよう、`-RenderOffscreen` を指定して CARLA を実行します。
- レンダリングコストを下げるには、`-quality-level=Low` を指定して CARLA を実行します。最終評価では使用しないでください。
- `-graphicsadapter=[id]` を指定し、CARLA とモデルを別々の GPU で実行します。

</details>

#### ベンチマークルートを手動で走破できますか？

```bash
python leaderboard/leaderboard/leaderboard_evaluator.py \
  --agent ${WORK_DIR}/leaderboard/leaderboard/autoagents/human_agent_keyboard.py \
  --routes ${WORK_DIR}/fail2drive_split/Generalization_PedestriansOnRoad_1085.xml
```

#### PDM-Lite エキスパートポリシーの実行

```bash
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent ${WORK_DIR}/team_code/visu_agent.py \
  --track MAP \
  --routes ${WORK_DIR}/fail2drive_split/Generalization_PedestriansOnRoad_1085.xml
```

#### TransFuser++ モデルの実行

モデルを実行する前に、チェックポイントを `checkpoints` フォルダーへダウンロードします。

```bash
mkdir -p checkpoints/tfpp
wget -P checkpoints/tfpp \
  https://huggingface.co/SimonGer/TFv5/resolve/main/all_towns/model_0030_0.pth \
  https://huggingface.co/SimonGer/TFv5/resolve/main/all_towns/config.json
```

続いて `LIVE_VISU` フラグを指定して TransFuser++ を実行し、モデルの入力をリアルタイムで確認します。

```bash
LIVE_VISU=1 python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --routes ${WORK_DIR}/fail2drive_split/Generalization_PedestriansOnRoad_1085.xml \
  --agent ${WORK_DIR}/team_code/sensor_agent.py \
  --agent-config ${WORK_DIR}/checkpoints/tfpp
```

## 評価

### Fail2Drive のルール

オフラインベンチマークであるため、ユーザーは評価用の全ルートとアセットにアクセスできます。過学習を防ぎ、Fail2Drive の公平性を保つため、以下のルールを定めています。

1. **Fail2Drive シナリオでの学習は禁止です。** モデルの学習やファインチューニングに、Fail2Drive で導入されたルート、シナリオ定義、アセットを使用してはなりません。このベンチマークは、厳密にホールドアウトされたテストセットとして使用します。

2. **外部データによる事前学習は許可されます。** 大規模な実世界データセット、インターネット規模のマルチモーダルコーパス、基盤モデル、VLM/LLM バックボーンを用いた事前学習は許可されます。このような一般的な視覚・言語知識はモデルの事前知識と見なされ、ベンチマークへの違反には当たりません。

3. **リーダーボードへの登録。** 最終スコアは、[公式リーダーボードリポジトリ](https://github.com/SimonGer/fail2drive_leaderboard)へ Pull Request で提出することを推奨します。これにより、一貫した比較と透明性の高いベンチマーク評価が可能になります。詳細は[コントリビューションガイドライン](https://github.com/SimonGer/fail2drive_leaderboard?tab=contributing-ov-file)を参照してください。

### SLURM による評価

SLURM クラスターを使用して、ベンチマーク全体でモデルを評価するためのツールを提供しています。[slurm_evaluate.py](slurm_evaluate.py) スクリプトは、[eval_num_jobs.txt](eval_num_jobs.txt) で指定された上限までジョブを自動的に投入し、評価を監視します。このスクリプトを特定のクラスターとモデルに合わせるには、多少の修正が必要です。スクリプト内の `NOTE` および `TODO` コメントを確認してください。

> **注：** スクリプトを実行する前に、必ず conda 環境を有効化してください：`conda activate fail2drive`

評価中に問題が発生した場合は、お気軽に Issue を作成してください。

### 結果の生成

最終スコアを取得するには、[tools/f2d_result_parser.py](tools/f2d_result_parser.py) スクリプトを使用します。

```bash
python tools/f2d_result_parser.py /path/to/results --method MyMethod
```

## Fail2Drive ツールボックス

Fail2Drive が提供するカスタマイズ可能なシナリオと新規アセットを使用して、新しいルートを生成するツールを提供しています。

![ツールボックス](./assets/toolbox.png)

これらのツールのドキュメントは[こちら](toolbox)にあります。コミュニティが作成したシナリオの共有や閲覧には、[シナリオハブ](https://github.com/SimonGer/fail2drive_scenario_hub)をご利用ください。

## 謝辞

本プロジェクトは、優れたオープンソースプロジェクトの成果の上に成り立っています。[carla_garage](https://github.com/autonomousvision/carla_garage) と [carla_route_generator](https://github.com/autonomousvision/carla_route_generator) を広範に活用し、[CARLA Leaderboard](https://github.com/carla-simulator/leaderboard) および [scenario_runner](https://github.com/carla-simulator/scenario_runner) フレームワークを統合しています。

また、本ベンチマークで評価した各モデルの作者にも感謝します：[SimLingo](https://github.com/RenzKa/simlingo)、[HiP-AD](https://github.com/nullmax-vision/HiP-AD)、[Orion](https://github.com/xiaomi-mlab/Orion)、[PlanT2](https://github.com/autonomousvision/plant2)、[Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo)。

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

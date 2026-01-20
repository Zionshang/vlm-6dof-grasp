# VLM-6DoF-Grasp

## 环境部署

以下命令均在仓库根目录执行。按顺序安装：economic_grasp → fastsam → vlm。

### 1) 安装 economic_grasp

依赖：Python 3.10 + CUDA 12.8（对应 Blackwell 架构 GPU，按需调整）。

1.1 PyTorch & CUDA

```bash
conda install openblas-devel -c anaconda
conda install -c nvidia/label/cuda-12.8.0 cuda-toolkit
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

1.2 MinkowskiEngine

```bash
cd economic_grasp/libs/MinkowskiEngine
conda install ninja
python setup.py install
cd -
```

1.3 Pip 依赖

```bash
pip install -r economic_grasp/requirements.txt
```

1.4 PointNet2

```bash
cd economic_grasp/libs/pointnet2
python setup.py install
cd -
```

1.5 KNN

```bash
cd economic_grasp/libs/knn
python setup.py install
cd -
```

1.6 GraspNetAPI

```bash
cd economic_grasp/libs/graspnetAPI
pip install .
cd -
```

### 2) 安装 fastsam

```bash
pip install -r fastsam/requirements.txt
```

### 3) 安装 vlm

先安装 ollama，并拉取模型：

```bash
# 安装 ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型（按需替换模型名，本项目使用的是）
ollama pull qwen3-vl:32b-instruct-q4_K_M
```

```bash
pip install -r vlm/requirements.txt
```

## 运行 Pipeline

入口脚本：`main_pipeline.py`

```bash
python main_pipeline.py \
  --data_dir example_data \
  --prompt "方盒子"
```

常用参数：
- `--prompt`：要检测的目标文本
- `--fastsam`：FastSAM 权重路径
- `--grasp_checkpoint`：EconomicGrasp 权重（不传则只做 VLM + 分割）
- `--use_sam`：是否使用 FastSAM（默认 True）
- `--use_collision`：是否做碰撞检测（默认 True）

# SAR图像多类别有向目标检测项目

本项目基于真实SAR图像数据集，实现多类别有向目标检测功能，支持检测ship、aircraft、car、tank、bridge和harbor等6类目标。

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 安装MMRotate框架

本项目基于MMRotate框架实现，请按照[官方文档](https://mmrotate.readthedocs.io/)安装MMRotate及相关依赖。

## 数据准备

项目已提供实际的SAR图像数据集，位于`data/train`和`data/test_A`目录下。

### 处理实际数据

使用以下命令处理实际数据集：

```bash
python scripts/prepare_real_data.py
```

参数说明：
- `--train_data_dir`：训练数据集目录，默认为`d:\编程项目\SAR\data\train`
- `--test_data_dir`：测试数据集目录，默认为`d:\编程项目\SAR\data\test_A`
- `--output_dir`：处理后数据保存目录，默认为`d:\编程项目\SAR\data\processed`
- `--val_ratio`：验证集比例，默认为0.2

数据处理完成后，会在`data/processed`目录下生成以下结构：
- `images/`：处理后的图像文件
- `annfiles/`：处理后的标注文件
- `ImageSets/`：数据集划分文件（train.txt、val.txt、test.txt）

## 模型训练

### 训练配置

项目提供了两种配置文件：
- `configs/sar_detector.py`：标准配置，使用ResNet50作为骨干网络
- `configs/sar_detector_light.py`：轻量级配置，使用ResNet34作为骨干网络，适合资源受限环境

### 启动训练

使用以下命令启动模型训练：

```bash
# 使用标准配置
python train.py --config configs/sar_detector.py

# 使用轻量级配置
python train.py --config configs/sar_detector_light.py
```

参数说明：
- `--config`：配置文件路径
- `--work-dir`：工作目录，用于保存日志和模型权重
- `--resume-from`：从指定的检查点恢复训练
- `--load-from`：加载预训练模型权重
- `--gpus`：使用的GPU数量
- `--seed`：随机种子

## 模型推理

使用训练好的模型进行推理：

```bash
python infer.py --config configs/sar_detector.py --checkpoint work_dirs/sar_detector/latest.pth --img-dir data/processed/test_A
```

参数说明：
- `--config`：配置文件路径
- `--checkpoint`：模型权重文件路径（必需）
- `--img-dir`：图像文件夹路径（必需）
- `--out-dir`：结果保存文件夹路径，默认为`results`
- `--score-thr`：置信度阈值，默认为0.3
- `--device`：运行设备，如cuda:0或cpu，默认为cuda:0

## 项目结构

```
SAR/
├── configs/             # 配置文件目录
│   ├── sar_detector.py           # 标准配置
│   └── sar_detector_light.py     # 轻量级配置
├── data/                # 数据目录
│   ├── train/           # 训练数据集
│   ├── test_A/          # 测试数据集
│   ├── processed/       # 处理后的数据
│   └── scripts/         # 数据处理脚本
├── scripts/             # 工具脚本
│   ├── prepare_data.py           # 通用数据准备脚本
│   └── prepare_real_data.py      # 实际数据准备脚本
├── src/                 # 源代码
├── tools/               # MMRotate工具
├── train.py             # 训练启动脚本
├── infer.py             # 推理脚本
└── requirements.txt     # 依赖列表
```

## 类别说明

本项目支持以下6类目标的检测：
1. ship（船只）
2. aircraft（飞机）
3. car（汽车）
4. tank（坦克）
5. bridge（桥梁）
6. harbor（港口）

## 注意事项

1. 训练过程中，模型权重和日志会保存在`work_dirs`目录下
2. 推理结果会保存在指定的`out_dir`目录下，包括可视化图像和检测结果文本文件
3. 如需调整模型参数，请修改相应的配置文件
4. 如有任何问题，请参考MMRotate官方文档或提交issue
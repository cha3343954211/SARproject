# SAR图像多类别有向目标检测项目

## 项目简介

本项目旨在实现大规模SAR图像中多类别有向目标的检测，支持船、飞机、汽车、油罐、港口、桥等6种典型类别的目标检测。项目基于深度学习框架，采用旋转目标检测算法，能够准确检测出SAR图像中任意方向的目标，并输出目标的类别、置信度和旋转边界框位置。

## 项目结构

```
SAR/
├── src/             # 源代码目录
├── configs/         # 配置文件目录
├── tools/           # 工具脚本目录
├── results/         # 结果输出目录
├── utils/           # 工具函数目录
├── scripts/         # 运行脚本目录
├── data/            # 数据集目录
└── README.md        # 项目说明文档
```

## 环境配置

### 依赖安装

本项目基于Python 3.7+和PyTorch框架，推荐使用以下方式安装依赖：

```bash
pip install -r requirements.txt
```

### 主要依赖库

- PyTorch
- MMCV
- MMRotate
- NumPy
- OpenCV
- scikit-learn
- matplotlib

## 数据集准备

### 数据格式

数据集应采用DOTA格式存储，每张图片对应一个标注文件（txt格式），标注文件与图片文件同名。

标注文件中每一行表示一个实例，格式为：`[x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, classname, difficulty]`

### 数据组织

数据集应按照以下结构组织：

```
data/
├── train/
│   ├── images/
│   └── labelTxt/
├── val/
│   ├── images/
│   └── labelTxt/
└── test/
    └── images/
```

## 使用方法

### 训练模型

```bash
python tools/train.py --config configs/sar_detector_config.py
```

### 测试模型

```bash
python tools/test.py --config configs/sar_detector_config.py --checkpoint results/latest.pth
```

### 推理预测

```bash
python tools/infer.py --config configs/sar_detector_config.py --checkpoint results/latest.pth --image_dir data/test/images --out_dir results/outputs
```

## 模型配置

在`configs/`目录下可以修改模型配置文件，调整网络结构、训练参数等。

## 结果评估

模型评估采用目标检测领域通用的mean Average Precision (mAP)指标。测试完成后，结果将保存在`results/`目录下。

## 注意事项

1. SAR图像具有特殊的成像特点，在预处理时需要进行针对性处理
2. 旋转目标检测与普通水平目标检测有所不同，需要特别注意旋转框的表示和计算
3. 训练模型时，建议使用GPU加速以提高训练效率

## 参考资料

1. Zhang, Xin, et al. "Rsar: Restricted state angle resolver and rotated sar benchmark." CVPR (2025).
2. Li, Yuxuan, et al. "SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection." NeurIPS Spotlight (2024).
3. MMRotate官方文档: https://mmrotate.readthedocs.io/
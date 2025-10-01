# SAR图像多类别有向目标检测项目快速入门指南

本指南将帮助您快速了解和使用SAR图像多类别有向目标检测项目。

## 1. 环境配置

### 1.1 安装依赖

首先，确保您的系统已安装Python 3.7或更高版本。然后，安装项目所需的依赖：

```bash
# 克隆项目（如果您还没有项目代码）
# git clone https://github.com/sar-detection-team/sar-object-detection.git
# cd sar-object-detection

# 安装依赖
pip install -r requirements.txt

# 可选：以开发模式安装项目
pip install -e .
```

### 1.2 安装MMRotate

本项目基于MMRotate框架，需要安装MMRotate及其依赖：

```bash
# 安装PyTorch和torchvision（根据您的CUDA版本选择合适的版本）
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 安装MMCV
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# 安装MMDetection
pip install mmdet

# 安装MMRotate
pip install mmrotate
```

## 2. 数据集准备

### 2.1 准备自己的数据集

将您的SAR图像数据集按照以下结构组织：

```
sar_dataset/
├── images/         # 存放图像文件
└── labelTxt/       # 存放标注文件
```

标注文件格式应遵循DOTA数据集格式：

```
x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
```

其中，(x1,y1)到(x4,y4)是目标的四个顶点坐标，category是目标类别，difficulty是目标难度级别。

### 2.2 使用数据准备脚本

使用提供的脚本预处理数据集：

```bash
python scripts/prepare_data.py --data-dir /path/to/your/sar/data --output-dir data/sar_dataset --resize --img-size 1024 --enhance
```

参数说明：
- `--data-dir`: 原始数据目录
- `--output-dir`: 输出数据目录
- `--resize`: 是否调整图像大小
- `--img-size`: 调整后的图像尺寸
- `--enhance`: 是否增强图像对比度

## 3. 训练模型

使用以下命令开始训练模型：

```bash
python tools/train.py configs/sar_detector.py --work-dir work_dirs/sar_detector
```

参数说明：
- `configs/sar_detector.py`: 配置文件路径
- `--work-dir`: 工作目录，用于保存模型检查点和日志

## 4. 测试模型

训练完成后，使用以下命令测试模型性能：

```bash
python tools/test.py configs/sar_detector.py work_dirs/sar_detector/latest.pth --eval mAP
```

参数说明：
- `configs/sar_detector.py`: 配置文件路径
- `work_dirs/sar_detector/latest.pth`: 模型检查点文件路径
- `--eval`: 评估指标

## 5. 推理和可视化

使用训练好的模型进行推理并可视化结果：

### 5.1 简单推理

```bash
python tools/inference.py configs/sar_detector.py work_dirs/sar_detector/latest.pth /path/to/image.jpg --out-dir results
```

参数说明：
- `configs/sar_detector.py`: 配置文件路径
- `work_dirs/sar_detector/latest.pth`: 模型检查点文件路径
- `/path/to/image.jpg`: 图像文件路径
- `--out-dir`: 输出结果保存目录

### 5.2 高级演示

使用演示脚本进行更详细的可视化：

```bash
python scripts/demo.py --config configs/sar_detector.py --checkpoint work_dirs/sar_detector/latest.pth --img /path/to/image.jpg --out-dir results --normalize --enhance
```

参数说明：
- `--config`: 配置文件路径
- `--checkpoint`: 模型检查点文件路径
- `--img`: 图像文件路径
- `--out-dir`: 输出结果保存目录
- `--normalize`: 是否归一化图像
- `--enhance`: 是否增强图像对比度

## 6. 自定义配置

### 6.1 修改模型配置

您可以修改`configs/sar_detector.py`文件来自定义模型配置，例如：
- 更改骨干网络（backbone）
- 调整学习率和优化器参数
- 修改数据增强策略
- 更改批量大小和训练轮数

### 6.2 支持的目标类别

当前项目支持6种目标类别：船舶（ship）、飞机（aircraft）、汽车（car）、坦克（tank）、桥梁（bridge）和港口（harbor）。您可以在`src/dataset.py`文件中修改`CLASSES`变量来调整支持的类别。

## 7. 常见问题解决

### 7.1 CUDA内存不足

如果遇到CUDA内存不足的问题，可以尝试以下解决方法：
- 减小批量大小（batch size）
- 使用较小的图像尺寸
- 减少数据增强的复杂度

### 7.2 训练不稳定

如果训练过程不稳定，可以尝试以下方法：
- 调整学习率
- 使用梯度裁剪（gradient clipping）
- 增加权重衰减（weight decay）

### 7.3 检测精度不高

如果检测精度不理想，可以尝试以下方法：
- 增加训练数据量
- 使用更强大的骨干网络
- 调整数据增强策略
- 尝试不同的检测头

## 8. 项目结构说明

```
sar-object-detection/
├── src/                # 源代码目录
│   └── dataset.py      # 数据集处理模块
├── configs/            # 配置文件目录
│   └── sar_detector.py # 模型配置文件
├── tools/              # 工具脚本目录
│   ├── train.py        # 训练脚本
│   ├── test.py         # 测试脚本
│   └── inference.py    # 推理脚本
├── utils/              # 工具函数目录
│   └── tools.py        # 常用工具函数
├── scripts/            # 辅助脚本目录
│   ├── prepare_data.py # 数据准备脚本
│   └── demo.py         # 演示脚本
├── data/               # 数据目录
├── results/            # 结果保存目录
├── work_dirs/          # 工作目录
├── README.md           # 项目说明文档
├── requirements.txt    # 依赖列表
├── setup.py            # 安装配置文件
└── quick_start.md      # 快速入门指南
```

## 9. 联系我们

如果您在使用过程中遇到任何问题，请随时联系我们：
- Email: sar_detection@example.com
- GitHub: https://github.com/sar-detection-team/sar-object-detection/issues
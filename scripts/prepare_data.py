import os
import sys
import argparse
import shutil
import numpy as np
import cv2
import json
import glob
from tqdm import tqdm


def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description='准备SAR图像数据集')
    parser.add_argument('--data-dir', required=True, help='原始数据目录')
    parser.add_argument('--output-dir', required=True, help='输出数据目录')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='训练集和验证集的划分比例')
    parser.add_argument('--img-size', type=int, default=1024, help='调整后的图像尺寸')
    parser.add_argument('--resize', action='store_true', help='是否调整图像大小')
    parser.add_argument('--enhance', action='store_true', help='是否增强图像对比度')
    args = parser.parse_args()
    return args


def create_directory_structure(output_dir):
    """创建数据集目录结构
    
    Args:
        output_dir (str): 输出目录路径
    """
    # 创建主目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子目录
    subdirs = [
        'images',
        'labelTxt',
        'ImageSets'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)


def collect_images_and_labels(data_dir):
    """收集图像和标注文件
    
    Args:
        data_dir (str): 数据目录
        
    Returns:
        list: 图像和标注文件列表
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 查找所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(data_dir, '*', f'*{ext}')))
    
    # 查找对应的标注文件
    data_pairs = []
    for img_file in image_files:
        # 获取图像文件名（不包含扩展名）
        img_name = os.path.splitext(os.path.basename(img_file))[0]
        
        # 尝试不同的标注文件格式
        label_file = None
        for ext in ['.txt', '.xml']:
            possible_label_file = os.path.join(data_dir, 'labelTxt', f'{img_name}{ext}')
            if os.path.exists(possible_label_file):
                label_file = possible_label_file
                break
        
        # 如果找不到标注文件，使用空标注
        if label_file is None:
            print(f'警告: 找不到图像 {img_name} 的标注文件')
        
        data_pairs.append((img_file, label_file))
    
    return data_pairs


def process_image(img_file, output_dir, img_size=None, enhance=False):
    """处理图像
    
    Args:
        img_file (str): 图像文件路径
        output_dir (str): 输出目录
        img_size (int, optional): 调整后的图像尺寸
        enhance (bool, optional): 是否增强图像对比度
        
    Returns:
        str: 处理后的图像文件路径
    """
    # 读取图像
    img = cv2.imread(img_file)
    if img is None:
        raise FileNotFoundError(f'无法读取图像: {img_file}')
    
    # 调整图像大小
    if img_size is not None:
        h, w = img.shape[:2]
        scale = min(img_size / h, img_size / w)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    
    # 增强图像对比度
    if enhance:
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 对图像的每个通道应用CLAHE
        if len(img.shape) == 3:
            enhanced_channels = []
            for i in range(3):
                enhanced_channel = clahe.apply(img[:, :, i])
                enhanced_channels.append(enhanced_channel)
            img = cv2.merge(enhanced_channels)
        else:
            img = clahe.apply(img)
    
    # 保存处理后的图像
    img_name = os.path.basename(img_file)
    output_path = os.path.join(output_dir, 'images', img_name)
    cv2.imwrite(output_path, img)
    
    return output_path


def process_label(label_file, output_dir, img_scale=1.0):
    """处理标注文件
    
    Args:
        label_file (str): 标注文件路径
        output_dir (str): 输出目录
        img_scale (float): 图像缩放比例
        
    Returns:
        str: 处理后的标注文件路径
    """
    if label_file is None:
        # 创建空标注文件
        label_name = os.path.splitext(os.path.basename(label_file))[0] + '.txt'
        output_path = os.path.join(output_dir, 'labelTxt', label_name)
        open(output_path, 'w').close()
        return output_path
    
    # 读取标注文件
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理标注
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 解析标注行
        parts = line.split()
        if len(parts) < 9:
            continue
        
        # 处理多边形顶点坐标
        poly = list(map(float, parts[:8]))
        # 应用缩放
        scaled_poly = [x * img_scale for x in poly]
        
        # 重新组合标注行
        processed_line = ' '.join(map(str, scaled_poly)) + ' ' + ' '.join(parts[8:])
        processed_lines.append(processed_line)
    
    # 保存处理后的标注文件
    label_name = os.path.basename(label_file)
    output_path = os.path.join(output_dir, 'labelTxt', label_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')
    
    return output_path


def split_dataset(data_pairs, split_ratio, output_dir):
    """划分数据集
    
    Args:
        data_pairs (list): 图像和标注文件对列表
        split_ratio (float): 训练集比例
        output_dir (str): 输出目录
    """
    # 随机打乱数据
    np.random.shuffle(data_pairs)
    
    # 计算划分点
    split_point = int(len(data_pairs) * split_ratio)
    
    # 划分数据集
    train_pairs = data_pairs[:split_point]
    val_pairs = data_pairs[split_point:]
    
    # 保存数据集分割信息
    def save_split_info(pairs, split_name):
        split_file = os.path.join(output_dir, 'ImageSets', f'{split_name}.txt')
        with open(split_file, 'w', encoding='utf-8') as f:
            for img_file, _ in pairs:
                img_name = os.path.splitext(os.path.basename(img_file))[0]
                f.write(img_name + '\n')
    
    save_split_info(train_pairs, 'train')
    save_split_info(val_pairs, 'val')
    
    return train_pairs, val_pairs


def create_dataset_files(output_dir):
    """创建数据集文件
    
    Args:
        output_dir (str): 输出目录
    """
    # 创建train.txt文件
    train_file = os.path.join(output_dir, 'train.txt')
    train_images = []
    
    with open(os.path.join(output_dir, 'ImageSets', 'train.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                train_images.append(line)
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for img_name in train_images:
            f.write(img_name + '\n')
    
    # 创建val.txt文件
    val_file = os.path.join(output_dir, 'val.txt')
    val_images = []
    
    with open(os.path.join(output_dir, 'ImageSets', 'val.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                val_images.append(line)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for img_name in val_images:
            f.write(img_name + '\n')


def main():
    """主函数"""
    args = parse_args()
    
    # 创建目录结构
    print('创建数据集目录结构...')
    create_directory_structure(args.output_dir)
    
    # 收集图像和标注文件
    print('收集图像和标注文件...')
    data_pairs = collect_images_and_labels(args.data_dir)
    print(f'共收集到 {len(data_pairs)} 个数据样本')
    
    # 处理图像和标注
    print('处理图像和标注...')
    processed_pairs = []
    
    for img_file, label_file in tqdm(data_pairs, desc='处理进度'):
        # 处理图像
        img_scale = 1.0
        if args.resize:
            # 读取原始图像尺寸
            img = cv2.imread(img_file)
            h, w = img.shape[:2]
            # 计算缩放比例
            scale = min(args.img_size / h, args.img_size / w)
            img_scale = scale
        
        # 处理图像
        processed_img = process_image(
            img_file,
            args.output_dir,
            img_size=args.img_size if args.resize else None,
            enhance=args.enhance
        )
        
        # 处理标注
        processed_label = process_label(
            label_file,
            args.output_dir,
            img_scale=img_scale
        )
        
        processed_pairs.append((processed_img, processed_label))
    
    # 划分数据集
    print('划分数据集...')
    train_pairs, val_pairs = split_dataset(processed_pairs, args.split_ratio, args.output_dir)
    print(f'训练集样本数: {len(train_pairs)}，验证集样本数: {len(val_pairs)}')
    
    # 创建数据集文件
    print('创建数据集文件...')
    create_dataset_files(args.output_dir)
    
    print('数据集准备完成！')


if __name__ == '__main__':
    main()
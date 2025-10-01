import os
import shutil
import random
import argparse

"""
SAR图像目标检测数据准备脚本
用于处理实际的SAR图像数据集
"""

def parse_args():
    parser = argparse.ArgumentParser(description='准备SAR图像目标检测数据')
    parser.add_argument('--train_data_dir', type=str, default='d:\\编程项目\\SAR\\data\\train',
                        help='训练数据集目录')
    parser.add_argument('--test_data_dir', type=str, default='d:\\编程项目\\SAR\\data\\test_A',
                        help='测试数据集目录')
    parser.add_argument('--output_dir', type=str, default='d:\\编程项目\\SAR\\data\\processed',
                        help='处理后数据保存目录')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例')
    return parser.parse_args()

def create_directory_structure(output_dir):
    """创建处理后的数据目录结构"""
    dirs = [
        os.path.join(output_dir, 'images'),
        os.path.join(output_dir, 'annfiles'),
        os.path.join(output_dir, 'ImageSets')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"已创建目录结构: {dirs}")

def collect_images_and_labels(data_dir):
    """收集图像和标注文件路径"""
    image_dir = os.path.join(data_dir, 'images')
    ann_dir = os.path.join(data_dir, 'annfiles')
    
    image_files = []
    ann_files = []
    
    # 遍历图像目录
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.png'):
            base_name = os.path.splitext(file_name)[0]
            image_path = os.path.join(image_dir, file_name)
            ann_path = os.path.join(ann_dir, f'{base_name}.txt')
            
            if os.path.exists(ann_path):
                image_files.append(image_path)
                ann_files.append(ann_path)
            else:
                print(f"警告: 未找到{file_name}对应的标注文件")
    
    return image_files, ann_files

def split_dataset(image_files, ann_files, val_ratio):
    """将数据集划分为训练集和验证集"""
    # 打乱数据
    combined = list(zip(image_files, ann_files))
    random.shuffle(combined)
    image_files[:], ann_files[:] = zip(*combined)
    
    # 划分数据集
    split_idx = int(len(image_files) * (1 - val_ratio))
    train_images = image_files[:split_idx]
    train_anns = ann_files[:split_idx]
    val_images = image_files[split_idx:]
    val_anns = ann_files[split_idx:]
    
    print(f"数据集划分完成: 训练集{len(train_images)}张, 验证集{len(val_images)}张")
    return train_images, train_anns, val_images, val_anns

def copy_files(file_list, dest_dir):
    """复制文件到目标目录"""
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.copy(file_path, dest_path)

def create_dataset_files(train_images, val_images, output_dir):
    """创建数据集文件列表"""
    # 创建训练集文件列表
    train_list_path = os.path.join(output_dir, 'ImageSets', 'train.txt')
    with open(train_list_path, 'w') as f:
        for image_path in train_images:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            f.write(f'{base_name}\n')
    
    # 创建验证集文件列表
    val_list_path = os.path.join(output_dir, 'ImageSets', 'val.txt')
    with open(val_list_path, 'w') as f:
        for image_path in val_images:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            f.write(f'{base_name}\n')
    
    print(f"已创建数据集文件列表: {train_list_path}, {val_list_path}")

def main():
    args = parse_args()
    
    # 创建目录结构
    create_directory_structure(args.output_dir)
    
    # 收集训练数据
    print("收集训练数据集...")
    train_image_files, train_ann_files = collect_images_and_labels(args.train_data_dir)
    print(f"已收集{len(train_image_files)}张训练图像及对应的标注文件")
    
    # 划分训练集和验证集
    train_images, train_anns, val_images, val_anns = split_dataset(
        train_image_files, train_ann_files, args.val_ratio
    )
    
    # 复制文件到处理后的数据目录
    print("复制训练集文件...")
    copy_files(train_images, os.path.join(args.output_dir, 'images'))
    copy_files(train_anns, os.path.join(args.output_dir, 'annfiles'))
    
    print("复制验证集文件...")
    # 验证集图像和标注也复制到相同目录
    copy_files(val_images, os.path.join(args.output_dir, 'images'))
    copy_files(val_anns, os.path.join(args.output_dir, 'annfiles'))
    
    # 创建数据集文件列表
    create_dataset_files(train_images, val_images, args.output_dir)
    
    # 处理测试数据
    print("收集测试数据集...")
    test_image_files, _ = collect_images_and_labels(args.test_data_dir)
    print(f"已收集{len(test_image_files)}张测试图像")
    
    # 复制测试图像
    test_output_dir = os.path.join(args.output_dir, 'test_A')
    os.makedirs(test_output_dir, exist_ok=True)
    copy_files(test_image_files, test_output_dir)
    
    # 创建测试集文件列表
    test_list_path = os.path.join(args.output_dir, 'ImageSets', 'test.txt')
    with open(test_list_path, 'w') as f:
        for image_path in test_image_files:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            f.write(f'{base_name}\n')
    
    print("\n数据准备完成！")
    print(f"处理后的数据保存在: {args.output_dir}")
    print(f"- 训练集: {len(train_images)}张图像")
    print(f"- 验证集: {len(val_images)}张图像")
    print(f"- 测试集: {len(test_image_files)}张图像")

if __name__ == '__main__':
    main()
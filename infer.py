import os
import argparse
import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

"""
SAR图像目标检测推理脚本
用于使用训练好的模型进行目标检测预测
"""

def parse_args():
    parser = argparse.ArgumentParser(description='SAR图像目标检测推理')
    parser.add_argument('--config', type=str, default='configs/sar_detector.py',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--img-dir', type=str, required=True,
                        help='图像文件夹路径')
    parser.add_argument('--out-dir', type=str, default='results',
                        help='结果保存文件夹路径')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='置信度阈值')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='运行设备，如cuda:0或cpu')
    return parser.parse_args()

def init_model(config_file, checkpoint_file, device):
    """初始化模型"""
    model = init_detector(config_file, checkpoint_file, device=device)
    return model

def process_image(model, img_path, out_dir, score_thr):
    """处理单张图像"""
    # 推理
    result = inference_detector(model, img_path)
    
    # 获取图像名称
    img_name = os.path.basename(img_path)
    
    # 保存可视化结果
    out_img_path = os.path.join(out_dir, img_name)
    model.show_result(
        img_path,
        result,
        score_thr=score_thr,
        bbox_color=(0, 255, 0),
        text_color=(0, 255, 0),
        thickness=2,
        font_size=10,
        out_file=out_img_path
    )
    
    # 保存检测结果到文本文件
    out_txt_path = os.path.join(out_dir, os.path.splitext(img_name)[0] + '.txt')
    save_detection_results(result, out_txt_path, score_thr, model.CLASSES)
    
    return out_img_path

def save_detection_results(result, out_file, score_thr, class_names):
    """保存检测结果到文本文件"""
    with open(out_file, 'w') as f:
        # 获取检测框和得分
        for i, bboxes in enumerate(result):
            if len(bboxes) > 0:
                for bbox in bboxes:
                    # 对于旋转目标检测，bbox通常包含8个坐标值（4个顶点）
                    if len(bbox) >= 9:  # 8个坐标 + 1个得分
                        # 检查置信度是否大于阈值
                        if bbox[-1] >= score_thr:
                            # 写入坐标、类别和得分
                            coords = ' '.join([str(round(c, 2)) for c in bbox[:8]])
                            f.write(f'{coords} {class_names[i]} {round(bbox[-1], 4)}\n')


def main():
    args = parse_args()
    
    # 检查文件和目录是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 {args.config} 不存在")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型权重文件 {args.checkpoint} 不存在")
        return
    
    if not os.path.exists(args.img_dir):
        print(f"错误: 图像文件夹 {args.img_dir} 不存在")
        return
    
    # 创建结果保存目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 初始化模型
    print("初始化模型...")
    model = init_model(args.config, args.checkpoint, args.device)
    
    # 获取图像列表
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = []
    for file_name in os.listdir(args.img_dir):
        if any(file_name.lower().endswith(ext) for ext in img_extensions):
            img_files.append(os.path.join(args.img_dir, file_name))
    
    if len(img_files) == 0:
        print(f"警告: 在 {args.img_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(img_files)} 张图像，开始处理...")
    
    # 处理每张图像
    for i, img_path in enumerate(img_files):
        print(f"处理图像 {i+1}/{len(img_files)}: {os.path.basename(img_path)}")
        out_img_path = process_image(model, img_path, args.out_dir, args.score_thr)
        print(f"  结果已保存至: {out_img_path}")
    
    print("\n所有图像处理完成！")
    print(f"结果保存在: {args.out_dir}")

if __name__ == '__main__':
    main()
import os
import sys
import argparse
import mmcv
import cv2
import numpy as np
import torch
from mmcv import Config
from mmdet.apis import inference_detector, init_detector
from mmrotate.core import obb2poly_np, poly2obb_np


def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description='SAR图像目标检测推理')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型检查点文件路径')
    parser.add_argument('img', help='图像文件或文件夹路径')
    parser.add_argument('--out-dir', help='输出结果保存目录')
    parser.add_argument('--score-thr', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--device', default='cuda:0', help='运行设备')
    args = parser.parse_args()
    return args


def draw_rotated_bbox(img, bbox, label, score, color=(0, 255, 0), thickness=2):
    """在图像上绘制旋转边界框
    
    Args:
        img (numpy.ndarray): 输入图像
        bbox (list): 旋转边界框参数 [x, y, w, h, angle]
        label (str): 类别标签
        score (float): 置信度分数
        color (tuple): 边界框颜色
        thickness (int): 边界框线宽
        
    Returns:
        numpy.ndarray: 绘制了边界框的图像
    """
    # 将旋转边界框转换为多边形
    poly = obb2poly_np(bbox, score=score)
    poly = poly[:8].reshape(4, 2)
    
    # 绘制多边形
    pts = poly.astype(np.int32)
    cv2.polylines(img, [pts], True, color, thickness)
    
    # 绘制标签和置信度
    text = f'{label}: {score:.2f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # 获取文本尺寸
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    # 计算文本框位置（左上角）
    text_x = int(poly[0][0])
    text_y = int(poly[0][1]) - 5
    
    # 确保文本在图像范围内
    if text_y < 20:
        text_y = int(poly[0][1]) + text_size[1] + 5
    
    # 绘制文本背景
    cv2.rectangle(
        img,
        (text_x, text_y - text_size[1] - 5),
        (text_x + text_size[0], text_y + 5),
        color,
        -1
    )
    
    # 绘制文本
    cv2.putText(
        img,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        font_thickness
    )
    
    return img


def inference(args):
    """执行推理
    
    Args:
        args (argparse.Namespace): 命令行参数
    """
    # 初始化模型
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    # 获取图像列表
    if os.path.isdir(args.img):
        img_files = [
            os.path.join(args.img, f) 
            for f in os.listdir(args.img) 
            if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
        ]
    else:
        img_files = [args.img]
    
    # 创建输出目录
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
    
    # 对每张图像执行推理
    for img_file in img_files:
        print(f'处理图像: {img_file}')
        
        # 加载图像
        img = mmcv.imread(img_file)
        
        # 执行推理
        result = inference_detector(model, img)
        
        # 解析结果
        bboxes = []
        labels = []
        scores = []
        
        for label_idx, bbox_list in enumerate(result):
            if len(bbox_list) > 0:
                for bbox in bbox_list:
                    # 检查置信度阈值
                    if bbox[-1] >= args.score_thr:
                        bboxes.append(bbox[:5])  # [x, y, w, h, angle]
                        labels.append(label_idx)
                        scores.append(bbox[-1])
        
        # 绘制检测结果
        result_img = img.copy()
        for bbox, label_idx, score in zip(bboxes, labels, scores):
            label = model.CLASSES[label_idx]
            result_img = draw_rotated_bbox(result_img, bbox, label, score)
        
        # 保存结果
        if args.out_dir is not None:
            # 获取图像文件名
            img_name = os.path.basename(img_file)
            out_file = os.path.join(args.out_dir, f'result_{img_name}')
            
            # 保存图像
            mmcv.imwrite(result_img, out_file)
            print(f'结果已保存到: {out_file}')
        
        # 显示结果
        else:
            cv2.imshow('SAR目标检测结果', result_img)
            cv2.waitKey(0)
    
    # 如果显示了结果，关闭所有窗口
    if args.out_dir is None:
        cv2.destroyAllWindows()


def main():
    """主函数"""
    args = parse_args()
    inference(args)


if __name__ == '__main__':
    main()
import os
import sys
import argparse
import cv2
import numpy as np
import mmcv
from mmcv import Config
from mmdet.apis import inference_detector, init_detector
from mmrotate.core import obb2poly_np
import matplotlib.pyplot as plt
from utils.tools import read_sar_image, normalize_sar_image, enhance_sar_image

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description='SAR图像目标检测演示')
    parser.add_argument('--config', default='configs/sar_detector.py', help='配置文件路径')
    parser.add_argument('--checkpoint', required=True, help='模型检查点文件路径')
    parser.add_argument('--img', required=True, help='图像文件路径')
    parser.add_argument('--out-dir', help='输出结果保存目录')
    parser.add_argument('--score-thr', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--device', default='cuda:0', help='运行设备')
    parser.add_argument('--normalize', action='store_true', help='是否归一化图像')
    parser.add_argument('--enhance', action='store_true', help='是否增强图像对比度')
    args = parser.parse_args()
    return args


def draw_detection_result(img, result, class_names, score_thr=0.3):
    """绘制检测结果
    
    Args:
        img (numpy.ndarray): 输入图像
        result (list): 检测结果
        class_names (list): 类别名称列表
        score_thr (float): 置信度阈值
        
    Returns:
        numpy.ndarray: 绘制了检测结果的图像
    """
    # 创建副本以避免修改原图
    result_img = img.copy()
    
    # 生成不同类别的颜色
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names))).tolist()
    colors = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors]
    
    # 处理每个类别的检测结果
    for label_idx, bboxes in enumerate(result):
        if len(bboxes) > 0:
            for bbox in bboxes:
                # 获取旋转框参数和置信度
                x, y, w, h, angle, score = bbox
                
                # 过滤低置信度结果
                if score < score_thr:
                    continue
                
                # 获取类别名称和颜色
                class_name = class_names[label_idx]
                color = colors[label_idx]
                
                # 将旋转边界框转换为多边形
                poly = obb2poly_np(bbox[:5], score=score)
                poly = poly[:8].reshape(4, 2)
                
                # 绘制多边形
                pts = poly.astype(np.int32)
                cv2.polylines(result_img, [pts], True, color, 2)
                
                # 绘制标签和置信度
                text = f'{class_name}: {score:.2f}'
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
                    result_img,
                    (text_x, text_y - text_size[1] - 5),
                    (text_x + text_size[0], text_y + 5),
                    color,
                    -1
                )
                
                # 绘制文本
                cv2.putText(
                    result_img,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    font_thickness
                )
    
    return result_img


def visualize_results(original_img, processed_img, result_img):
    """可视化原始图像、处理后的图像和检测结果
    
    Args:
        original_img (numpy.ndarray): 原始图像
        processed_img (numpy.ndarray): 处理后的图像
        result_img (numpy.ndarray): 检测结果图像
    """
    # 创建一个3列的图像显示
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示原始图像
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示处理后的图像
    axes[1].imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('处理后的图像')
    axes[1].axis('off')
    
    # 显示检测结果
    axes[2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('检测结果')
    axes[2].axis('off')
    
    # 调整布局
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    cfg = Config.fromfile(args.config)
    
    # 初始化模型
    print('初始化模型...')
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    # 读取图像
    print(f'读取图像: {args.img}')
    original_img = read_sar_image(args.img)
    
    # 处理图像
    processed_img = original_img.copy()
    
    # 归一化图像
    if args.normalize:
        processed_img = normalize_sar_image(processed_img)
    
    # 增强图像对比度
    if args.enhance:
        processed_img = enhance_sar_image(processed_img)
    
    # 执行推理
    print('执行推理...')
    result = inference_detector(model, processed_img)
    
    # 绘制检测结果
    print('绘制检测结果...')
    result_img = draw_detection_result(
        processed_img,
        result,
        model.CLASSES,
        score_thr=args.score_thr
    )
    
    # 保存结果
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        # 获取图像文件名
        img_name = os.path.basename(args.img)
        # 保存原始图像
        original_out_file = os.path.join(args.out_dir, f'original_{img_name}')
        cv2.imwrite(original_out_file, original_img)
        # 保存处理后的图像
        processed_out_file = os.path.join(args.out_dir, f'processed_{img_name}')
        cv2.imwrite(processed_out_file, processed_img)
        # 保存检测结果
        result_out_file = os.path.join(args.out_dir, f'result_{img_name}')
        cv2.imwrite(result_out_file, result_img)
        print(f'结果已保存到: {args.out_dir}')
    
    # 可视化结果
    visualize_results(original_img, processed_img, result_img)


if __name__ == '__main__':
    main()
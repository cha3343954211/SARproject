import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmrotate.core import poly2obb, obb2poly


def read_sar_image(img_path):
    """读取SAR图像
    
    Args:
        img_path (str): 图像文件路径
        
    Returns:
        numpy.ndarray: 读取的图像数据
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'无法读取图像: {img_path}')
    # 转换为三通道图像以便后续处理
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def normalize_sar_image(img, min_val=0, max_val=255):
    """归一化SAR图像
    
    Args:
        img (numpy.ndarray): 输入图像
        min_val (int): 归一化后的最小值
        max_val (int): 归一化后的最大值
        
    Returns:
        numpy.ndarray: 归一化后的图像
    """
    # 计算图像的最小值和最大值
    img_min = np.min(img)
    img_max = np.max(img)
    
    # 防止除零错误
    if img_max - img_min == 0:
        return img
    
    # 归一化
    normalized_img = ((img - img_min) / (img_max - img_min)) * (max_val - min_val) + min_val
    normalized_img = normalized_img.astype(np.uint8)
    return normalized_img


def enhance_sar_image(img, clahe_clip_limit=2.0, clahe_grid_size=(8, 8)):
    """增强SAR图像对比度
    
    Args:
        img (numpy.ndarray): 输入图像
        clahe_clip_limit (float): CLAHE的裁剪限制
        clahe_grid_size (tuple): CLAHE的网格大小
        
    Returns:
        numpy.ndarray: 增强后的图像
    """
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
    
    # 对图像的每个通道应用CLAHE
    if len(img.shape) == 3:
        enhanced_channels = []
        for i in range(3):
            enhanced_channel = clahe.apply(img[:, :, i])
            enhanced_channels.append(enhanced_channel)
        enhanced_img = cv2.merge(enhanced_channels)
    else:
        enhanced_img = clahe.apply(img)
    
    return enhanced_img


def rotate_image(img, angle, center=None, scale=1.0):
    """旋转图像
    
    Args:
        img (numpy.ndarray): 输入图像
        angle (float): 旋转角度
        center (tuple): 旋转中心
        scale (float): 缩放比例
        
    Returns:
        numpy.ndarray: 旋转后的图像
    """
    # 获取图像尺寸
    (h, w) = img.shape[:2]
    
    # 如果未指定旋转中心，使用图像中心
    if center is None:
        center = (w / 2, h / 2)
    
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 执行旋转
    rotated_img = cv2.warpAffine(img, M, (w, h))
    
    return rotated_img


def resize_sar_image(img, size, keep_ratio=True):
    """调整SAR图像大小
    
    Args:
        img (numpy.ndarray): 输入图像
        size (tuple): 目标大小 (width, height)
        keep_ratio (bool): 是否保持宽高比
        
    Returns:
        numpy.ndarray: 调整大小后的图像
    """
    if keep_ratio:
        # 计算缩放比例
        h, w = img.shape[:2]
        scale = min(size[1] / h, size[0] / w)
        new_size = (int(w * scale), int(h * scale))
        
        # 调整图像大小
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        
        # 创建目标大小的画布
        result_img = np.zeros((size[1], size[0], 3), dtype=np.uint8) if len(img.shape) == 3 else np.zeros((size[1], size[0]), dtype=np.uint8)
        
        # 计算居中放置的位置
        x_offset = (size[0] - new_size[0]) // 2
        y_offset = (size[1] - new_size[1]) // 2
        
        # 放置调整后的图像
        result_img[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = resized_img
        
        return result_img
    else:
        # 直接调整大小
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def visualize_detection_result(img, bboxes, labels, scores, class_names, score_thr=0.3):
    """可视化检测结果
    
    Args:
        img (numpy.ndarray): 输入图像
        bboxes (list): 边界框列表 [x, y, w, h, angle]
        labels (list): 类别标签列表
        scores (list): 置信度分数列表
        class_names (list): 类别名称列表
        score_thr (float): 置信度阈值
        
    Returns:
        numpy.ndarray: 可视化后的图像
    """
    # 创建副本以避免修改原图
    result_img = img.copy()
    
    # 生成不同类别的颜色
    colors = generate_colors(len(class_names))
    
    # 绘制每个检测结果
    for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
        # 过滤低置信度结果
        if score < score_thr:
            continue
        
        # 获取类别名称和颜色
        class_name = class_names[label]
        color = colors[label]
        
        # 将旋转边界框转换为多边形
        poly = obb2poly(bbox, score=score)
        poly = poly[:8].reshape(4, 2).astype(np.int32)
        
        # 绘制多边形
        cv2.polylines(result_img, [poly], True, color, 2)
        
        # 绘制标签和置信度
        text = f'{class_name}: {score:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # 获取文本尺寸
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        # 计算文本框位置
        text_x = poly[0][0]
        text_y = poly[0][1] - 5
        
        # 确保文本在图像范围内
        if text_y < 20:
            text_y = poly[0][1] + text_size[1] + 5
        
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


def generate_colors(num_colors):
    """生成指定数量的颜色
    
    Args:
        num_colors (int): 需要的颜色数量
        
    Returns:
        list: 颜色列表 [(R, G, B), ...]
    """
    colors = []
    for i in range(num_colors):
        # 使用HSV颜色空间，确保颜色分布均匀
        hue = i / num_colors
        rgb = plt.cm.hsv(hue)[:3]  # 获取RGB值
        # 转换为0-255范围
        rgb = tuple(int(c * 255) for c in rgb)
        colors.append(rgb)
    return colors


def save_detection_result(img, bboxes, labels, scores, class_names, save_path, score_thr=0.3):
    """保存检测结果
    
    Args:
        img (numpy.ndarray): 输入图像
        bboxes (list): 边界框列表
        labels (list): 类别标签列表
        scores (list): 置信度分数列表
        class_names (list): 类别名称列表
        save_path (str): 保存路径
        score_thr (float): 置信度阈值
    """
    # 可视化结果
    result_img = visualize_detection_result(img, bboxes, labels, scores, class_names, score_thr)
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图像
    cv2.imwrite(save_path, result_img)


def convert_dota_to_poly(dota_label_file):
    """将DOTA格式的标注转换为多边形格式
    
    Args:
        dota_label_file (str): DOTA格式的标注文件路径
        
    Returns:
        list: 多边形标注列表
    """
    polygons = []
    with open(dota_label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            # 提取多边形顶点
            poly = list(map(float, parts[:8]))
            # 提取类别
            category = parts[8]
            # 提取难度级别
            difficulty = int(parts[9]) if len(parts) > 9 else 0
            polygons.append({
                'poly': poly,
                'category': category,
                'difficulty': difficulty
            })
    return polygons


def convert_poly_to_obb(polygons):
    """将多边形转换为旋转边界框
    
    Args:
        polygons (list): 多边形标注列表
        
    Returns:
        list: 旋转边界框列表
    """
    obbs = []
    for poly_info in polygons:
        poly = poly_info['poly']
        # 转换为旋转边界框
        obb = poly2obb(poly, code=1)
        obbs.append({
            'obb': obb,
            'category': poly_info['category'],
            'difficulty': poly_info['difficulty']
        })
    return obbs


def calculate_iou(bbox1, bbox2):
    """计算两个旋转边界框的交并比
    
    Args:
        bbox1 (list): 第一个旋转边界框 [x, y, w, h, angle]
        bbox2 (list): 第二个旋转边界框 [x, y, w, h, angle]
        
    Returns:
        float: 交并比
    """
    # 将旋转边界框转换为多边形
    poly1 = obb2poly(bbox1)
    poly2 = obb2poly(bbox2)
    
    # 计算多边形的面积
    area1 = cv2.contourArea(poly1[:8].reshape(-1, 2))
    area2 = cv2.contourArea(poly2[:8].reshape(-1, 2))
    
    # 计算交集面积
    # 创建两个多边形的掩码
    h = max(int(max(poly1[1::2])), int(max(poly2[1::2]))) + 10
    w = max(int(max(poly1[0::2])), int(max(poly2[0::2]))) + 10
    
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)
    
    # 填充多边形
    pts1 = np.array(poly1[:8]).reshape(-1, 2).astype(np.int32)
    pts2 = np.array(poly2[:8]).reshape(-1, 2).astype(np.int32)
    
    cv2.fillPoly(mask1, [pts1], 1)
    cv2.fillPoly(mask2, [pts2], 1)
    
    # 计算交集
    intersection = np.logical_and(mask1, mask2).sum()
    
    # 计算并集
    union = area1 + area2 - intersection
    
    # 计算交并比
    if union == 0:
        return 0
    
    iou = intersection / union
    return iou
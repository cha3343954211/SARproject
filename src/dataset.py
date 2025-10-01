import os
import cv2
import numpy as np
import mmcv
from mmdet.datasets import CustomDataset
from mmrotate.core import poly2obb

class SARDataset(CustomDataset):
    """SAR图像数据集类，用于加载和处理SAR图像数据集。"""
    
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
    
    def __init__(self, 
                 ann_file, 
                 img_prefix, 
                 pipeline, 
                 classes=None, 
                 data_root=None, 
                 img_suffix='.png', 
                 seg_prefix=None, 
                 proposal_file=None, 
                 test_mode=False, 
                 filter_empty_gt=True):
        """初始化SAR数据集
        
        Args:
            ann_file (str): 标注文件路径
            img_prefix (str): 图像文件前缀路径
            pipeline (list): 数据处理流水线
            classes (tuple[str]): 类别名称元组
            data_root (str): 数据根目录
            img_suffix (str): 图像文件后缀
            seg_prefix (str): 分割文件前缀
            proposal_file (str): 候选区域文件
            test_mode (bool): 是否为测试模式
            filter_empty_gt (bool): 是否过滤没有目标的图像
        """
        super(SARDataset, self).__init__(
            ann_file=ann_file,
            img_prefix=img_prefix,
            pipeline=pipeline,
            classes=classes,
            data_root=data_root,
            img_suffix=img_suffix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt)
    
    def load_annotations(self, ann_file):
        """加载标注信息
        
        Args:
            ann_file (str): 标注文件路径
            
        Returns:
            list[dict]: 图像标注信息列表
        """
        
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        
        for img_id in img_ids:
            # 构造图像路径
            img_path = os.path.join(self.img_prefix, f'{img_id}{self.img_suffix}')
            
            # 构造标注文件路径
            ann_path = os.path.join(self.img_prefix, 'labelTxt', f'{img_id}.txt')
            
            # 检查文件是否存在
            if not os.path.exists(img_path):
                continue
            
            # 读取图像尺寸
            img = mmcv.imread(img_path)
            height, width = img.shape[:2]
            
            # 加载标注信息
            data_info = {
                'filename': f'{img_id}{self.img_suffix}',
                'ann': {
                    'bboxes': [],
                    'labels': [],
                    'bboxes_ignore': [],
                    'labels_ignore': []
                },
                'height': height,
                'width': width
            }
            
            # 如果是测试模式，不需要加载标注
            if self.test_mode:
                data_infos.append(data_info)
                continue
            
            # 读取标注文件
            if os.path.exists(ann_path):
                with open(ann_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析标注行
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    
                    # 获取四个顶点坐标
                    poly = np.array(parts[:8], dtype=np.float32)
                    # 获取类别名称
                    class_name = parts[8]
                    # 获取难度级别
                    difficulty = int(parts[9]) if len(parts) > 9 else 0
                    
                    # 检查类别是否在支持的类别列表中
                    if class_name not in self.CLASSES:
                        continue
                    
                    # 将多边形转换为旋转边界框
                    obb = poly2obb(poly, code=1)
                    
                    # 计算边界框面积
                    area = obb[2] * obb[3]
                    
                    # 如果面积太小，视为无效目标
                    if area < 10:
                        continue
                    
                    # 获取类别索引
                    label = self.CLASSES.index(class_name)
                    
                    # 根据难度级别决定是否忽略
                    if difficulty == 1:
                        data_info['ann']['bboxes_ignore'].append(obb)
                        data_info['ann']['labels_ignore'].append(label)
                    else:
                        data_info['ann']['bboxes'].append(obb)
                        data_info['ann']['labels'].append(label)
            
            # 转换为numpy数组
            if data_info['ann']['bboxes']:
                data_info['ann']['bboxes'] = np.array(data_info['ann']['bboxes'], dtype=np.float32)
                data_info['ann']['labels'] = np.array(data_info['ann']['labels'], dtype=np.int64)
            
            if data_info['ann']['bboxes_ignore']:
                data_info['ann']['bboxes_ignore'] = np.array(data_info['ann']['bboxes_ignore'], dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(data_info['ann']['labels_ignore'], dtype=np.int64)
            
            # 过滤没有目标的图像
            if self.filter_empty_gt and not data_info['ann']['bboxes'].size:
                continue
            
            data_infos.append(data_info)
        
        return data_infos
    
    def _filter_imgs(self, min_size=32):
        """过滤过小的图像
        
        Args:
            min_size (int): 最小图像尺寸
            
        Returns:
            list[int]: 过滤后的图像索引列表
        """
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """获取指定索引的标注信息
        
        Args:
            idx (int): 数据索引
            
        Returns:
            dict: 标注信息字典
        """
        return self.data_infos[idx]['ann']

    def get_cat_ids(self, idx):
        """获取指定索引的类别ID
        
        Args:
            idx (int): 数据索引
            
        Returns:
            list[int]: 类别ID列表
        """
        return self.data_infos[idx]['ann']['labels'].tolist()

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """格式化模型输出结果
        
        Args:
            results (list): 模型输出结果列表
            jsonfile_prefix (str): JSON文件前缀
            
        Returns:
            list: 格式化后的结果列表
        """
        # 按照赛题要求的格式整理结果
        formatted_results = []
        
        for idx, result in enumerate(results):
            img_info = self.data_infos[idx]
            filename = img_info['filename']
            
            # 创建当前图像的结果字典
            img_result = {
                'image': filename,
                'poly': [],
                'scores': [],
                'labels': []
            }
            
            # 处理每个类别的检测结果
            for label_idx, bboxes in enumerate(result):
                if bboxes.size == 0:
                    continue
                
                for bbox in bboxes:
                    # 获取旋转框参数
                    x, y, w, h, angle = bbox[:5]
                    # 计算置信度
                    score = bbox[5]
                    # 获取类别名称
                    label = self.CLASSES[label_idx]
                    
                    # 将旋转框转换为多边形
                    poly = self.obb2poly([x, y, w, h, angle])
                    
                    # 添加到结果中
                    img_result['poly'].append(poly)
                    img_result['scores'].append(float(score))
                    img_result['labels'].append(label)
            
            # 转换为numpy数组
            if img_result['poly']:
                img_result['poly'] = np.array(img_result['poly'], dtype=np.float32)
            else:
                img_result['poly'] = np.zeros((0, 8), dtype=np.float32)
            
            formatted_results.append(img_result)
        
        # 如果提供了JSON文件前缀，保存结果
        if jsonfile_prefix is not None:
            os.makedirs(jsonfile_prefix, exist_ok=True)
            result_file = os.path.join(jsonfile_prefix, 'results.pkl')
            np.save(result_file, formatted_results)
        
        return formatted_results
    
    @staticmethod
    def obb2poly(obb):
        """将旋转边界框转换为多边形
        
        Args:
            obb (list): 旋转边界框参数 [x, y, w, h, angle]
            
        Returns:
            list: 多边形顶点坐标 [x1, y1, x2, y2, x3, y3, x4, y4]
        """
        x, y, w, h, angle = obb
        
        # 计算旋转矩阵
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # 计算四个顶点相对于中心的坐标
        dx = [-w/2, -w/2, w/2, w/2]
        dy = [-h/2, h/2, h/2, -h/2]
        
        # 应用旋转并平移
        poly = []
        for i in range(4):
            px = x + dx[i] * cos_angle - dy[i] * sin_angle
            py = y + dx[i] * sin_angle + dy[i] * cos_angle
            poly.extend([px, py])
        
        return poly
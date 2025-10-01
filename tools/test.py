import os
import sys
import argparse
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmrotate.models import build_detector
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import json


def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description='测试SAR图像目标检测模型')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型检查点文件路径')
    parser.add_argument(
        '--out',
        default=None,
        help='输出结果文件路径')
    parser.add_argument(
        '--json_out',
        default=None,
        help='输出JSON格式结果文件路径')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['mAP'],
        help='评估指标，例如：mAP, recall')
    parser.add_argument(
        '--show',
        action='store_true',
        help='是否显示检测结果')
    parser.add_argument(
        '--show_dir',
        default=None,
        help='检测结果保存目录')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='是否使用GPU收集结果')
    parser.add_argument(
        '--tmpdir',
        help='分布式测试时的临时目录')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='分布式测试启动器')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 检查输出文件路径
    assert args.out or args.show or args.json_out, \
        ('至少需要指定一个选项: --out 保存结果, --show 显示结果, ' \
         '--json_out 保存JSON格式结果')
    
    # 如果同时指定了--out和--json_out，确保它们不冲突
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('输出文件必须是.pkl或.pickle格式')
    
    # 加载配置文件
    cfg = Config.fromfile(args.config)
    
    # 设置分布式测试
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    
    # 构建数据集
    dataset = build_dataset(cfg.data.test)
    
    # 构建数据加载器
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    # 构建模型
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 加载模型检查点
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # 为模型添加类别信息（如果存在于检查点中）
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    # 设置模型为评估模式
    model.eval()
    
    # 执行测试
    if not distributed:
        # 单GPU测试
        outputs = single_gpu_test(
            model,
            data_loader,
            args.show,
            args.show_dir,
            show_score_thr=0.3)
    else:
        # 多GPU测试
        outputs = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect)
    
    # 获取分布式信息
    rank, _ = get_dist_info()
    
    # 如果是主进程，处理结果
    if rank == 0:
        # 保存结果到文件
        if args.out:
            print(f'将结果保存到 {args.out}')
            mmcv.dump(outputs, args.out)
        
        # 保存JSON格式结果
        if args.json_out:
            print(f'将JSON格式结果保存到 {args.json_out}')
            
            # 格式化结果
            json_results = []
            for idx, result in enumerate(outputs):
                img_id = dataset.data_infos[idx]['filename']
                
                # 处理每个类别的检测结果
                for label_idx, bboxes in enumerate(result):
                    if bboxes.size == 0:
                        continue
                    
                    for bbox in bboxes:
                        # 获取检测框参数和置信度
                        x, y, w, h, angle, score = bbox
                        
                        # 创建结果字典
                        result_dict = {
                            'image_id': img_id,
                            'category_id': label_idx + 1,
                            'bbox': [float(x), float(y), float(w), float(h), float(angle)],
                            'score': float(score)
                        }
                        json_results.append(result_dict)
            
            # 保存为JSON文件
            with open(args.json_out, 'w') as f:
                json.dump(json_results, f)
        
        # 评估模型性能
        if args.eval:
            eval_results = dataset.evaluate(outputs, args.eval, **cfg.evaluation)
            for k, v in eval_results.items():
                print(f'{k}: {v}')


if __name__ == '__main__':
    main()
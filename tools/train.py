import os
import sys
import argparse
import mmcv
from mmcv import Config
from mmcv.runner import set_random_seed
import torch
from mmdet.apis import train_detector
from mmrotate.models import build_detector
from src.dataset import SARDataset


def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description='训练SAR图像目标检测模型')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--work-dir', help='工作目录')
    parser.add_argument('--resume-from', help='从指定检查点恢复训练')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='训练过程中不进行验证')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='GPU数量')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='设置是否使用确定性算法')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    cfg = Config.fromfile(args.config)
    
    # 自定义工作目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # 如果配置中没有指定工作目录，根据配置文件名自动创建
        cfg.work_dir = os.path.join('./work_dirs', 
                                  os.path.splitext(os.path.basename(args.config))[0])
    
    # 从指定检查点恢复训练
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
        
    # 设置GPU数量
    cfg.gpu_ids = range(args.gpus)
    
    # 创建工作目录
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    
    # 将配置写入文件
    cfg.dump(os.path.join(cfg.work_dir, 'config.py'))
    
    # 设置随机种子
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    
    # 构建模型
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    
    # 构建数据集
    datasets = [SARDataset(**cfg.data['train'])]
    
    # 是否在训练过程中进行验证
    if cfg.checkpoint_config is not None:
        # 保存配置信息到检查点
        cfg.checkpoint_config.meta = dict(
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    
    # 训练模型
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=(not args.no_validate),
        timestamp=mmcv.timestamp())


if __name__ == '__main__':
    main()
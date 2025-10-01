import os
import argparse
import sys

"""
SAR图像目标检测训练启动脚本
用于启动MMRotate框架下的模型训练
"""

def parse_args():
    parser = argparse.ArgumentParser(description='SAR图像目标检测模型训练')
    parser.add_argument('--config', type=str, default='configs/sar_detector.py',
                        help='配置文件路径')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='工作目录，用于保存日志和模型权重')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='从指定的检查点恢复训练')
    parser.add_argument('--load-from', type=str, default=None,
                        help='加载预训练模型权重')
    parser.add_argument('--gpus', type=int, default=1,
                        help='使用的GPU数量')
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子')
    parser.add_argument('--deterministic', action='store_true',
                        help='是否设置确定性选项')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 {args.config} 不存在")
        sys.exit(1)
    
    # 构建命令行参数
    cmd = [
        'python', '-m', 'mmrotate.tools.train',
        args.config
    ]
    
    # 添加可选参数
    if args.work_dir is not None:
        cmd.extend(['--work-dir', args.work_dir])
    
    if args.resume_from is not None:
        cmd.extend(['--resume-from', args.resume_from])
    
    if args.load_from is not None:
        cmd.extend(['--load-from', args.load_from])
    
    if args.gpus != 1:
        cmd.extend(['--gpus', str(args.gpus)])
    
    if args.seed != 0:
        cmd.extend(['--seed', str(args.seed)])
    
    if args.deterministic:
        cmd.append('--deterministic')
    
    # 打印命令
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行命令
    os.system(' '.join(cmd))

if __name__ == '__main__':
    main()
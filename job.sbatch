#!/bin/bash
#SBATCH --job-name=rsgnn           # 指定作业名称
#SBATCH --output=output.txt        # 指定作业输出的文件名
#SBATCH --error=error.txt          # 指定错误输出的文件名数
#SBATCH --time=60:00:00     # 指定作业执行的最长时间
#SBATCH --partition=gpu_all
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
python3 main.py --m Random --dataset cifar10 --c 1 --rsgnn_epochs 40
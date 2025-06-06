#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -J seg
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err

#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH  --constraint=[a100]
##SBATCH --mem=30G



#Go to your working directory
[ ! -d "slurm_logs" ] && echo "Create a directory slurm_logs" && mkdir -p slurm_logs

#Module load the desired application if necessary
module load cuda/11.8 #Always check the module needed on the login node “module avail”
echo "===> load cuda/11.8"
source ~/.bashrc
conda activate deepvortex

nvidia-smi
nvcc --version
hostname
NUM_GPU_AVAILABLE=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
echo "NUM_GPU_AVAILABLE:"
echo $NUM_GPU_AVAILABLE


#Edit below with the launching command:
# cfg=$1
# PY_ARGS=${@:2}
# python train.py --cfg $cfg ${PY_ARGS}
python train.py --config config/segmentation/pathline_transformer.yaml
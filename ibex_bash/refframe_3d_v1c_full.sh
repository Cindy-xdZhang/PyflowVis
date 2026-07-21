#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-11%12
#SBATCH -J RF3Dfull
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint=a100
#SBATCH --mem=96G

# Verify_rft3d_1.1 part-1 (user 2026-07-21: "do 1, 2 and 3"): FULL-resolution
# deltaWing recheck of the s2x2 conclusion (statistical tie, E/E0 criterion).
# 5% tier at the two candidate peak lrs; baseline width solves to m~260 -> a100
# only (V100 would blow the 6h wall on 64k steps at that width) and 10h cap.
#   {bl, ctM1} x lr {5e-4, 3e-4} x 3 seeds = 12 tasks
# Full-res data in float64 + dvdt peaks ~12 GB host RAM -> 96G.
# Submit:  sbatch ibex_bash/refframe_3d_v1c_full.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

i=$SLURM_ARRAY_TASK_ID
SEEDS=(0 7777 1)
s=${SEEDS[$((i % 3))]}
combo=$((i / 3))
arm_idx=$((combo / 2))
LRS=(5e-4 3e-4); LR=${LRS[$((combo % 2))]}

if [ "$arm_idx" == "0" ]; then
  arm=bl; mode=baseline; obs=tvfull
else
  arm=ctM1; mode=pro_budget; obs=consttrans
fi

cd experiments/referenceframe_inr_3d || exit 1
export PYFLOWVIS_DATA3D=${PYFLOWVIS_DATA3D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA3D=$PYFLOWVIS_DATA3D"
mkdir -p outputs

echo "=== Verify_rft3d_1.1 full-res: deltawing mlp arm=$arm lr=$LR f0.05 seed=$s ==="
python -u run_experiment3d.py --field deltawing \
    --model mlp --budget_frac 0.05 --d_base 10 \
    --tau=-1 --n_windows 1 --max_inrs 3 --observer "$obs" \
    --modes "$mode" --epochs 1000 --batch_size 32000 \
    --max_steps_per_epoch 64 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_rft3d_1.1/deltawing_full_mlp_${arm}_f0.05_lr${LR}_s${s}" || exit 1
echo "=== DONE task $i ==="

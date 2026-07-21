#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-11%12
#SBATCH -J RF3Dsiren
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_rft3d_1.1 part-2: SIREN (coordnet) side of the 3D scoreboard, mirroring
# the 2D two-architecture story.  deltaWing s2x2 5%, equal width m=66 (the
# CoordNet-skeleton parameter formula is architecture-invariant).  lr probed at
# the 2D compression-tier coordnet range {1e-4, 3e-4}; 3e-4 collapsed on 2D
# bouss/cyl (mean-flow attractor) -- watching for that failure mode IS part of
# the probe; recovery readout checks mse plateaus.
#   {bl, ctM1} x lr {1e-4, 3e-4} x 3 seeds = 12 tasks
# Submit:  sbatch ibex_bash/refframe_3d_v1d_siren.sh

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
LRS=(1e-4 3e-4); LR=${LRS[$((combo % 2))]}

if [ "$arm_idx" == "0" ]; then
  arm=bl; mode=baseline; obs=tvfull
else
  arm=ctM1; mode=pro_budget; obs=consttrans
fi

cd experiments/referenceframe_inr_3d || exit 1
export PYFLOWVIS_DATA3D=${PYFLOWVIS_DATA3D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA3D=$PYFLOWVIS_DATA3D"
mkdir -p outputs

echo "=== Verify_rft3d_1.1 siren: deltawing_s2x2 coordnet arm=$arm lr=$LR f0.05 seed=$s ==="
python -u run_experiment3d.py --field deltawing --stride_t 2 --stride_xyz 2 \
    --model coordnet --budget_frac 0.05 --d_base 10 \
    --tau=-1 --n_windows 1 --max_inrs 3 --observer "$obs" \
    --modes "$mode" --epochs 1000 --batch_size 32000 \
    --max_steps_per_epoch 64 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_rft3d_1.1/deltawing_s2x2_coordnet_${arm}_f0.05_lr${LR}_s${s}" || exit 1
echo "=== DONE task $i ==="

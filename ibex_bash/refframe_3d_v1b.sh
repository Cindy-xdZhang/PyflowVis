#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-35%36
#SBATCH -J RF3Dv1b
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_rft3d_1.1 wave 2 (user 2026-07-21: "deploy 1 and 2; queueing is fine
# so deploy liberally").  Wave-1 found the deltaWing lr ladder INVERTED vs 2D
# (5e-4 peak, higher lr monotonically worse, low end unexplored) and a 5%-tier
# statistical tie (+0.03).  This wave pins the peak AND fills the scoreboard in
# one go so tier readings don't need a redo if the peak moves to 3e-4:
#   A (tasks 0-11):  5% tier, lr probe DOWN: {bl, ctM1} x lr {3e-4, 2e-4} x 3
#   B (tasks 12-35): tier fill: {bl, ctM1} x frac {2.5%, 10%} x lr {5e-4, 3e-4}
#                    x 3 seeds
# ctM1 = single-window single-region consttrans (tau=-1), equal-width to bl.
# Submit:  sbatch ibex_bash/refframe_3d_v1b.sh

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

if [ "$combo" -lt 4 ]; then
  # A: 5% tier, low-lr probe
  frac=0.05
  arm_idx=$((combo / 2))
  LRS=(3e-4 2e-4); LR=${LRS[$((combo % 2))]}
else
  # B: tier fill at the two candidate peaks
  cb=$((combo - 4))
  arm_idx=$((cb / 4))
  rem=$((cb % 4))
  FRACS=(0.025 0.10); frac=${FRACS[$((rem / 2))]}
  LRS=(5e-4 3e-4); LR=${LRS[$((rem % 2))]}
fi

if [ "$arm_idx" == "0" ]; then
  arm=bl; mode=baseline; obs=tvfull
else
  arm=ctM1; mode=pro_budget; obs=consttrans
fi

cd experiments/referenceframe_inr_3d || exit 1
export PYFLOWVIS_DATA3D=${PYFLOWVIS_DATA3D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA3D=$PYFLOWVIS_DATA3D"
mkdir -p outputs

echo "=== Verify_rft3d_1.1 wave2: deltawing_s2x2 mlp arm=$arm lr=$LR f$frac seed=$s ==="
python -u run_experiment3d.py --field deltawing --stride_t 2 --stride_xyz 2 \
    --model mlp --budget_frac "$frac" --d_base 10 \
    --tau=-1 --n_windows 1 --max_inrs 3 --observer "$obs" \
    --modes "$mode" --epochs 1000 --batch_size 32000 \
    --max_steps_per_epoch 64 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_rft3d_1.1/deltawing_s2x2_mlp_${arm}_f${frac}_lr${LR}_s${s}" || exit 1
echo "=== DONE task $i ==="

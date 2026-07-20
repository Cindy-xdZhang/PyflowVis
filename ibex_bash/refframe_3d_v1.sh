#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-17%18
#SBATCH -J RF3Dv1
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_rft3d_1.1 (user 2026-07-20: "deploy plan A"): deltaWing s2x2 5%
# compression scoreboard, first 3D wave.  E/E0 prior says weak applicability
# (global observer explains ~30%), so the experiment doubles as the 3D
# extrapolation test of the E/E0 criterion.  mlp only (the 2D RFT-benefiting
# architecture); lr probed symmetrically (2D lesson: lr was the single biggest
# lever, and the mlp ladder topped out at 1.5-2e-3 on cylinder):
#   {bl, ctM1} x lr {5e-4, 1e-3, 2e-3} x 3 seeds {0,7777,1} = 18 tasks
# ctM1 = single-window single-region consttrans (tau=-1 fast path), byte-exact
# side info 63 B.  Scale protocol: stride 2/2 dataset variant (~121 MiB),
# max_steps_per_epoch=64 (64k steps total, same order as the 2D recipe).
# Submit:  sbatch ibex_bash/refframe_3d_v1.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

i=$SLURM_ARRAY_TASK_ID
SEEDS=(0 7777 1)
LRS=(5e-4 1e-3 2e-3)
s=${SEEDS[$((i % 3))]}
combo=$((i / 3))
arm_idx=$((combo / 3)); LR=${LRS[$((combo % 3))]}

if [ "$arm_idx" == "0" ]; then
  arm=bl; mode=baseline; obs=tvfull
else
  arm=ctM1; mode=pro_budget; obs=consttrans
fi

cd experiments/referenceframe_inr_3d || exit 1
export PYFLOWVIS_DATA3D=${PYFLOWVIS_DATA3D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA3D=$PYFLOWVIS_DATA3D"
mkdir -p outputs

echo "=== Verify_rft3d_1.1: deltawing_s2x2 mlp arm=$arm lr=$LR f0.05 seed=$s ==="
python -u run_experiment3d.py --field deltawing --stride_t 2 --stride_xyz 2 \
    --model mlp --budget_frac 0.05 --d_base 10 \
    --tau=-1 --n_windows 1 --max_inrs 3 --observer "$obs" \
    --modes "$mode" --epochs 1000 --batch_size 32000 \
    --max_steps_per_epoch 64 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_rft3d_1.1/deltawing_s2x2_mlp_${arm}_f0.05_lr${LR}_s${s}" || exit 1
echo "=== DONE task $i ==="

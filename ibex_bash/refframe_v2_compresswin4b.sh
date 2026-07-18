#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-1%2
#SBATCH -J RFv2win4b
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_compresswin_1.4 wave 2 -- the LAST 2 tasks of the user's 24 budget.
# Wave-1: M=1 ct closed most of the cylinder x mlp gap (-4~-7 -> -0.3~-0.9)
# but lost at every tier; residual cause: a single Galilean frame freezes the
# advecting street but UN-freezes the lab-steady obstacle/formation zone (the
# 216-px region the tau-merge itself refuses to merge at tau=0.02).
# Wave-2 bet at the closest tier (5%, gap -0.34): M=2 obstacle-separate
# (nw=1, tau=0.02, absorb=128 -> [23800, 216] px after absorb), per-region
# consttrans observers (wake auto-solves ~(0.816,0), obstacle auto-solves ~0),
# NEW --alloc capsmall (par.4.4h "proportional + capacity floor": obstacle
# capped at 1 param/sample -> m=8, wake takes the rest -> m=28 ~ bl m=29).
# 2 seeds {0, 7777} (third seed has no budget; mlp spreads here are 0.5-1.6).
# Submit:  sbatch ibex_bash/refframe_v2_compresswin4b.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

SEEDS=(0 7777)
s=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd experiments/referenceframe_inr_v2 || exit 1
export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"
mkdir -p outputs

echo "=== Verify_compresswin_1.4 wave2: cylinder2d mlp ct-M2-capsmall f0.05 seed=$s ==="
python -u run_experiment.py --field cylinder2d --model mlp \
    --budget_frac 0.05 --d_base 10 --tau 0.02 --absorb_min_pixels 128 \
    --n_windows 1 --allow_full_window --max_inrs 3 \
    --observer consttrans --alloc capsmall \
    --modes pro_budget --epochs 1000 --lr 3e-4 --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_compresswin_1.4/cylinder2d_mlp_ctM2_f0.05_lr3e-4_s${s}" || exit 1
echo "=== DONE task $SLURM_ARRAY_TASK_ID ==="

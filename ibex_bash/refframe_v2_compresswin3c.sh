#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-2%3
#SBATCH -J RFv2win3c
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_compresswin_1.3 wave 3 (final 3 tasks; session total 63/64): bouss 5%
# last shot. Wave-2 narrowed the gap to -0.48 (cf@7e-5+wu0.1 66.60 [66.44..66.82]
# vs bl@1e-4+wu0.1 67.08). cf@7e-5's three seeds converge tightly at 66.5-66.8
# (ceiling-limited, not noise) while cf@1e-4 splits (left tail even at wu0.2)
# -> probe the midpoint lr 8e-5 with the longer warmup. bl is not re-run at
# 8e-5: its grid brackets the point (7e-5+wu 65.31, 1e-4+wu 67.08) and cannot
# plausibly exceed its existing optimum there; pro only needs > 67.08 to win.
# Submit:  sbatch ibex_bash/refframe_v2_compresswin3c.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

SEEDS=( 0 7777 1 )
s=${SEEDS[$SLURM_ARRAY_TASK_ID]}
arm=cf; LR=8e-5; WU=0.2; frac=0.05
field=boussinesq; db=10; absorb=256; tau=0.5

cd experiments/referenceframe_inr_v2 || exit 1
export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"
mkdir -p outputs

echo "=== Verify_compresswin_1.3 wave3: arm=$arm constfull frac=$frac lr=$LR wu=$WU seed=$s ==="
python -u run_experiment.py --field "$field" --model coordnet \
    --budget_frac "$frac" --d_base "$db" --tau "$tau" \
    --absorb_min_pixels "$absorb" --n_windows 1 --allow_full_window --max_inrs 3 \
    --observer constfull --modes pro_budget --epochs 1000 --lr "$LR" --lr_final 1e-6 \
    --warmup_frac "$WU" --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_compresswin_1.3/${field}_${arm}_f${frac}_lr${LR}_wu${WU}_s${s}" || exit 1
echo "=== DONE task $SLURM_ARRAY_TASK_ID ==="

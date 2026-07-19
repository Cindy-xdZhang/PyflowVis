#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-23%24
#SBATCH -J RFv2win4d
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_compresswin_1.4 wave 4 (user 2026-07-19: "deploy both plans", 24 tasks).
# Wave-3 found the mlp lr ladder unfinished (1e-3 >> 5e-4 >> 3e-4, both arms)
# and the 5% tier turned positive at lr=1e-3 (ctM2 +0.27, 3/3 seeds paired win).
# Plan A (tasks 0-11): migrate lr=1e-3 to the two remaining scoreboard cells,
#   {bl, ctM2-capsmall} x {2.5%, 10%} x 3 seeds.  Old readings there are @3e-4
#   (-0.86 / -0.61) so both arms rerun at the new lr, best-vs-best stays fair.
# Plan B (tasks 12-23): continue the lr ladder at 5%,
#   {bl, ctM2-capsmall} x lr {1.5e-3, 2e-3} x 3 seeds.
# ctM2-capsmall = wave-2 winner structure (nw=1 tau=0.02 absorb=128 ->
# [23800,216]px, per-region consttrans, small region capped by --alloc capsmall).
# Submit:  sbatch ibex_bash/refframe_v2_compresswin4d.sh

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

field=cylinder2d; db=10
if [ "$combo" -lt 4 ]; then
  # Plan A: lr fixed 1e-3, sweep {arm} x {frac}
  LR=1e-3
  arm_idx=$((combo / 2))
  FRACS=(0.025 0.10); frac=${FRACS[$((combo % 2))]}
else
  # Plan B: frac fixed 0.05, sweep {arm} x {lr}
  frac=0.05
  cb=$((combo - 4))
  arm_idx=$((cb / 2))
  LRS=(1.5e-3 2e-3); LR=${LRS[$((cb % 2))]}
fi

if [ "$arm_idx" == "0" ]; then
  arm=bl; mode=baseline; obs=tvfull; nw=2; tau=0.02; absorb=128
  alloc=uniform; fullwin=""
else
  arm=ctM2; mode=pro_budget; obs=consttrans; nw=1; tau=0.02; absorb=128
  alloc=capsmall; fullwin="--allow_full_window"
fi

cd experiments/referenceframe_inr_v2 || exit 1
export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"
mkdir -p outputs

echo "=== Verify_compresswin_1.4 wave4: cylinder2d mlp arm=$arm lr=$LR f$frac seed=$s ==="
python -u run_experiment.py --field "$field" --model mlp \
    --budget_frac "$frac" --d_base "$db" --tau "$tau" \
    --absorb_min_pixels "$absorb" --n_windows "$nw" $fullwin --max_inrs 3 \
    --observer "$obs" --alloc "$alloc" \
    --modes "$mode" --epochs 1000 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_compresswin_1.4/cylinder2d_mlp_${arm}_f${frac}_lr${LR}_s${s}" || exit 1
echo "=== DONE task $i ==="

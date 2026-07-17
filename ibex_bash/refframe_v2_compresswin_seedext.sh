#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-17%18
#SBATCH -J RFv2wse
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_compresswin_1.1 SEED EXTENSION (docs par.4.4l): +3 seeds {1,2,3} for
# the decisive cells only, -> 5-seed means (with the {0,7777} runs of job
# 49000294) to settle the statistical ties / big-split cells:
#   boussinesq f0.05/f0.10 x {bl, w2M2} @ lr 1e-4  (10% was a -0.08 dB tie)
#   rfc        f0.05       x {bl, w1M1} @ lr 3e-4  (both arms seed-split)
# Task layout: 0-11 boussinesq: i = 6*frac_idx + 3*arm_idx + seed_idx
#              12-17 rfc:       j = i-12; arm_idx = j/3; seed_idx = j%3
# Submit:  sbatch ibex_bash/refframe_v2_compresswin_seedext.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

i=$SLURM_ARRAY_TASK_ID
SEEDS=(1 2 3)

if [ "$i" -lt 12 ]; then
  field=boussinesq; db=10; absorb=256; LR=1e-4
  FRACS=(0.05 0.10)
  frac=${FRACS[$((i / 6))]}
  rest=$((i % 6))
  if [ "$((rest / 3))" == "0" ]; then
    arm=bl; mode=baseline; nw=2; tau=0.5; fullwin=""
  else
    arm=w2M2; mode=pro_budget; nw=2; tau=0.5; fullwin=""
  fi
  s=${SEEDS[$((rest % 3))]}
else
  field=rfc; db=4; absorb=0; LR=3e-4; frac=0.05
  j=$((i - 12))
  if [ "$((j / 3))" == "0" ]; then
    arm=bl; mode=baseline; nw=2; tau=0.05; fullwin=""
  else
    arm=w1M1; mode=pro_budget; nw=1; tau=0.05; fullwin="--allow_full_window"
  fi
  s=${SEEDS[$((j % 3))]}
fi

export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"

cd experiments/referenceframe_inr_v2 || exit 1
mkdir -p outputs

echo "=== Verify_compresswin_1.1 seedext: field=$field arm=$arm mode=$mode frac=$frac"
echo "    nw=$nw tau=$tau absorb=$absorb d=$db lr=$LR epochs=1000 n_seeds=1 seed=$s ==="
python -u run_experiment.py --field "$field" --model coordnet \
    --budget_frac "$frac" --d_base "$db" --tau "$tau" \
    --absorb_min_pixels "$absorb" --n_windows "$nw" $fullwin --max_inrs 3 \
    --modes "$mode" --epochs 1000 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_compresswin_1.1/${field}_${arm}_f${frac}_lr${LR}_s${s}" || exit 1
echo "=== DONE task $i ==="

#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-35%32
#SBATCH -J RFv2c25
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# mainExp_compress_1.2: 2.5% byte-budget tier (CR >= 40x) on rfc / cylinder2d /
# boussinesq, under the CORRECTED protocol learned in Verify_compresswin_1.1
# (docs par.4.4l) -- NOT comparable to the mainExp_compress_1.1 tiers:
#   * M <= 3 hard rule (--max_inrs 3); per-field best partition structure:
#       rfc        pro = w1M1 (1 window, tau=0.05, absorb=0)   -> M=1, equal-m
#       cylinder2d pro = w2M2 (2 windows, tau=0.1, absorb=256) -> M=2 (dry-run)
#       boussinesq pro = w2M2 (2 windows, tau=0.5, absorb=256) -> M=2
#   * per-(field,arch) lr: coordnet swept 2 arms (rfc {1e-4,3e-4}; cylinder &
#     boussinesq {3e-5,1e-4} -- bouss 3e-4 collapses, cyl 3e-4 collapsed in
#     par.4.6); mlp fixed 3e-4 (stable everywhere). Baseline gets the same
#     lr arms as pro (best-vs-best fairness).
#   * epochs 1000 (user 2026-07-16), 2 seeds {0,7777} mean.
# Sizes (budget_calc, d frozen): rfc bl m=8 / pro m=8 (equal width);
# cylinder bl m=21 / pro m_r=14 x2; boussinesq bl m=17 / pro m_r=11 x2.
#
# Task layout (36):
#   0-23  coordnet: field=i/8; rest=i%8: mode=rest/4 (0=bl,1=pro),
#         lr_idx=(rest/2)%2, seed=rest%2
#   24-35 mlp:      j=i-24: field=j/4; rest=j%4: mode=rest/2, seed=rest%2
# Submit:  sbatch ibex_bash/refframe_v2_compress25.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

i=$SLURM_ARRAY_TASK_ID
FIELDS=(rfc cylinder2d boussinesq)
DB=(4 10 10)
TAUS=(0.05 0.1 0.5)
ABSORB=(0 256 256)
PRONW=(1 2 2)
LRA=(1e-4 3e-5 3e-5)
LRB=(3e-4 1e-4 1e-4)
SEEDS=(0 7777)
frac=0.025

if [ "$i" -lt 24 ]; then
  model=coordnet
  f=$((i / 8)); rest=$((i % 8))
  m_idx=$((rest / 4)); lr_idx=$(((rest / 2) % 2)); s=${SEEDS[$((rest % 2))]}
  if [ "$lr_idx" == "0" ]; then LR=${LRA[$f]}; else LR=${LRB[$f]}; fi
else
  model=mlp
  j=$((i - 24)); f=$((j / 4)); rest=$((j % 4))
  m_idx=$((rest / 2)); s=${SEEDS[$((rest % 2))]}
  LR=3e-4
fi

field=${FIELDS[$f]}; db=${DB[$f]}; tau=${TAUS[$f]}; absorb=${ABSORB[$f]}
if [ "$m_idx" == "0" ]; then
  mode=baseline; nw=2; fullwin=""
else
  mode=pro_budget; nw=${PRONW[$f]}
  fullwin=""; [ "$nw" == "1" ] && fullwin="--allow_full_window"
fi

export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"

cd experiments/referenceframe_inr_v2 || exit 1
mkdir -p outputs

echo "=== mainExp_compress_1.2: field=$field model=$model mode=$mode frac=$frac"
echo "    nw=$nw tau=$tau absorb=$absorb d=$db lr=$LR epochs=1000 n_seeds=1 seed=$s ==="
python -u run_experiment.py --field "$field" --model "$model" \
    --budget_frac "$frac" --d_base "$db" --tau "$tau" \
    --absorb_min_pixels "$absorb" --n_windows "$nw" $fullwin --max_inrs 3 \
    --modes "$mode" --epochs 1000 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/mainExp_compress_1.2/${field}_${model}_${mode}_f${frac}_lr${LR}_s${s}" || exit 1
echo "=== DONE task $i ==="

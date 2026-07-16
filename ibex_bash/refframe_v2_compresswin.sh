#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-119%32
#SBATCH -J RFv2win
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_compresswin_1.1: find a coordnet (SIREN) configuration where the
# proposed method (partition + reference-frame transform) BEATS its own
# baseline at strict compression budgets (5% / 10% of raw field bytes).
# User constraints (2026-07-16): total INR count 1-3 ONLY (tiny budgets must
# not be split further -- enforced via --max_inrs 3); epochs 1000 (converged
# performance is what matters, 2000 not needed); sweep tau AND lr, lr swept
# for BASELINE TOO so the comparison is best-vs-best fair.
# n_windows=1 arms deviate from the old ">=2 windows" spec rule on user
# instruction ("1-3 partitions"); windowing was measured as a 5-6 dB
# optimization tax (docs par.4.4b), M=1 concentrates the whole budget.
#
# boussinesq tau -> M dry-run (absorb=256, 2026-07-16):
#   1 window: tau=0.5 -> M=1   tau=0.4 -> M=2   tau=0.35 -> M=3
#   2 windows: tau=0.5 -> M=2 ([1,1], old operating point)   tau=0.4 -> M=3
# rfc: any tau -> N=1 per window; pro arm = 1 window tau=0.05 -> M=1
# (the par.4.4b equal-param observer-win config, closes the rfc coordnet gap).
#
# Task layout (120 tasks):
#   0-15    boussinesq baseline: i = 8*frac_idx + 2*lr_idx + seed_idx
#   16-95   boussinesq pro:      j=i-16; arm=j/16; rest=j%16 as baseline
#           arms: w1M1(nw=1,tau=.5) w1M2(1,.4) w1M3(1,.35) w2M2(2,.5) w2M3(2,.4)
#   96-119  rfc block:           j=i-96; mode=j/12 (0=bl, 1=pro w1M1 tau=.05);
#           rest=j%12: frac_idx=rest/6, lr_idx=(rest/2)%3, seed_idx=rest%2
# lr grids: boussinesq (1e-5 3e-5 1e-4 3e-4); rfc (1e-5 1e-4 3e-4)
# Submit:  sbatch ibex_bash/refframe_v2_compresswin.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

i=$SLURM_ARRAY_TASK_ID
FRACS=(0.05 0.10)
SEEDS=(0 7777)

if [ "$i" -lt 96 ]; then
  field=boussinesq; db=10; absorb=256
  LRS=(1e-5 3e-5 1e-4 3e-4)
  if [ "$i" -lt 16 ]; then
    arm=bl; mode=baseline; nw=2; tau=0.5; fullwin=""
    rest=$i
  else
    j=$((i - 16)); a=$((j / 16)); rest=$((j % 16))
    ARMNW=(1 1 1 2 2); ARMTAU=(0.5 0.4 0.35 0.5 0.4)
    ARMNAME=(w1M1 w1M2 w1M3 w2M2 w2M3)
    arm=${ARMNAME[$a]}; mode=pro_budget; nw=${ARMNW[$a]}; tau=${ARMTAU[$a]}
    fullwin=""; [ "$nw" == "1" ] && fullwin="--allow_full_window"
  fi
  frac=${FRACS[$((rest / 8))]}
  LR=${LRS[$(((rest / 2) % 4))]}
  s=${SEEDS[$((rest % 2))]}
else
  field=rfc; db=4; absorb=0
  LRS=(1e-5 1e-4 3e-4)
  j=$((i - 96)); rest=$((j % 12))
  if [ "$((j / 12))" == "0" ]; then
    arm=bl; mode=baseline; nw=2; tau=0.05; fullwin=""
  else
    arm=w1M1; mode=pro_budget; nw=1; tau=0.05; fullwin="--allow_full_window"
  fi
  frac=${FRACS[$((rest / 6))]}
  LR=${LRS[$(((rest / 2) % 3))]}
  s=${SEEDS[$((rest % 2))]}
fi

export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"

cd experiments/referenceframe_inr_v2 || exit 1
mkdir -p outputs

echo "=== Verify_compresswin_1.1: field=$field arm=$arm mode=$mode frac=$frac"
echo "    nw=$nw tau=$tau absorb=$absorb d=$db lr=$LR epochs=1000 n_seeds=1 seed=$s ==="
python -u run_experiment.py --field "$field" --model coordnet \
    --budget_frac "$frac" --d_base "$db" --tau "$tau" \
    --absorb_min_pixels "$absorb" --n_windows "$nw" $fullwin --max_inrs 3 \
    --modes "$mode" --epochs 1000 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_compresswin_1.1/${field}_${arm}_f${frac}_lr${LR}_s${s}" || exit 1
echo "=== DONE task $i ==="

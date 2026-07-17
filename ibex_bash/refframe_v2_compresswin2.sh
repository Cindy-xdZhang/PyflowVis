#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-119%32
#SBATCH -J RFv2win2
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_compresswin_1.2 (docs par.4.4l/4.4m follow-up): close the two open
# boussinesq cells of the user's acceptance criterion.
#   * bouss 5%: 5-seed revision flipped the 2-seed +0.73 win to -1.18 -- the
#     baseline has a bad-basin left tail at lr 1e-4 and pro w1M1 is unstable
#     exactly at 1e-4. Try the mid lrs {5e-5, 7e-5} for ALL arms (baseline
#     included, fairness) hoping w1M1/w2M2 stabilize near the bl optimum.
#   * bouss 2.5%: w2M2 lost -11.9 (2-window split leaves m_r=11 vs bl m=17).
#     Deploy the no-split w1M1 arm (M=1, m=16 ~ bl 17 near-equal width).
# Full grid, 5-seed protocol (bouss@1e-4 needs >=5 seeds, par.4.4l revision):
#   frac {0.025, 0.05} x lr {3e-5, 5e-5, 7e-5, 1e-4} x arm {bl, w1M1, w2M2}
#   x seeds {0, 7777, 1, 2, 3}  = 120 tasks; cells already run by jobs
#   49000294 / 49023725 / 49025811 are SKIPPED via metrics-json existence
#   checks (~26 skips), so ~94 actually train.
#   i = 60*frac_idx + 15*lr_idx + 5*arm_idx + seed_idx
# Submit:  sbatch ibex_bash/refframe_v2_compresswin2.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

i=$SLURM_ARRAY_TASK_ID
FRACS=(0.025 0.05)
LRS=(3e-5 5e-5 7e-5 1e-4)
ARMS=(bl w1M1 w2M2)
SEEDS=(0 7777 1 2 3)

frac=${FRACS[$((i / 60))]}
r=$((i % 60))
LR=${LRS[$((r / 15))]}
r2=$((r % 15))
arm=${ARMS[$((r2 / 5))]}
s=${SEEDS[$((r2 % 5))]}

field=boussinesq; db=10; absorb=256; tau=0.5
case "$arm" in
  bl)   mode=baseline;   nw=2; fullwin="" ;;
  w1M1) mode=pro_budget; nw=1; fullwin="--allow_full_window" ;;
  w2M2) mode=pro_budget; nw=2; fullwin="" ;;
esac

cd experiments/referenceframe_inr_v2 || exit 1

# skip cells already produced by earlier jobs (same config, same seed)
J1="outputs/Verify_compresswin_1.1/${field}_${arm}_f${frac}_lr${LR}_s${s}/${field}/${field}_metrics.json"
J2="outputs/mainExp_compress_1.2/${field}_coordnet_${mode}_f${frac}_lr${LR}_s${s}/${field}/${field}_metrics.json"
if [ -f "$J1" ] || [ -f "$J2" ]; then
  echo "=== SKIP task $i (already run): $J1 / $J2"; exit 0
fi

export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"
mkdir -p outputs

echo "=== Verify_compresswin_1.2: arm=$arm mode=$mode frac=$frac lr=$LR seed=$s"
echo "    nw=$nw tau=$tau absorb=$absorb d=$db epochs=1000 n_seeds=1 ==="
python -u run_experiment.py --field "$field" --model coordnet \
    --budget_frac "$frac" --d_base "$db" --tau "$tau" \
    --absorb_min_pixels "$absorb" --n_windows "$nw" $fullwin --max_inrs 3 \
    --modes "$mode" --epochs 1000 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_compresswin_1.1/${field}_${arm}_f${frac}_lr${LR}_s${s}" || exit 1
echo "=== DONE task $i ==="

#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-44%32
#SBATCH -J RFv2win3
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_compresswin_1.3 (docs par.4.4n): beat the coordnet baseline on the
# boussinesq 2.5% (and 5%) strict-compression cells with a single global
# observer (M=1, single window) plus three targeted fixes:
#   1. byte-accounting v2: N==1 stores no label map; const observers store
#      8-12 B instead of Tw*3*4 -> pro width m_r 16 -> 17 == bl width at 2.5%
#      (equal-width comparison, the structure that wins on rfc).
#   2. observer variants (dry-run evidence, diag_agent_observer_variants.py):
#      the bouss global observer is essentially a CONSTANT upward translation
#      (c ~ -0.035, b ~ +0.179; time variation explains only 0.1% more energy).
#      consttrans kills the rotation sweep (xi-bbox inflation 1.288 -> 1.111)
#      and the per-frame observer jitter. constfull / tvfull arms isolate the
#      contribution of each ingredient.
#   3. lr warmup (warmup_frac 0.1): targets the SIREN high-lr bad-basin left
#      tail (par.4.4l: bl AND pro seeds collapse 5-16 dB at lr >= 1e-4).
#      Applied symmetrically to baseline arms -- fairness.
# Grid: 15 combos x 3 seeds {0, 7777, 1} = 45 tasks (<= 64 budget, user
# 2026-07-17). Existing same-protocol cells (bl 2.5%@1e-4 wu0, bl 5%@{7e-5,
# 1e-4} wu0) are NOT re-run; they are cited from Verify_compresswin_1.1/1.2.
# Submit:  sbatch ibex_bash/refframe_v2_compresswin3.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

#        0      1      2      3      4      5      6      7      8      9     10     11     12     13     14
ARMS=(  bl     bl     bl     ct     ct     ct     cf     tv     tv     bl     bl     bl     ct     ct     ct  )
FRACS=( 0.025  0.025  0.025  0.025  0.025  0.025  0.025  0.025  0.025  0.05   0.05   0.05   0.05   0.05   0.05 )
LRS=(   1e-4   1.5e-4 1.5e-4 1e-4   1.5e-4 1e-4   1e-4   1e-4   1e-4   7e-5   1e-4   1.5e-4 7e-5   1e-4   1.5e-4)
WUS=(   0.1    0.1    0      0.1    0.1    0      0.1    0.1    0      0.1    0.1    0.1    0.1    0.1    0.1 )
SEEDS=( 0 7777 1 )

i=$SLURM_ARRAY_TASK_ID
combo=$((i / 3))
s=${SEEDS[$((i % 3))]}
arm=${ARMS[$combo]}
frac=${FRACS[$combo]}
LR=${LRS[$combo]}
WU=${WUS[$combo]}

field=boussinesq; db=10; absorb=256; tau=0.5
case "$arm" in
  bl) mode=baseline;   obs=tvfull;     nw=2; fullwin="" ;;
  ct) mode=pro_budget; obs=consttrans; nw=1; fullwin="--allow_full_window" ;;
  cf) mode=pro_budget; obs=constfull;  nw=1; fullwin="--allow_full_window" ;;
  tv) mode=pro_budget; obs=tvfull;     nw=1; fullwin="--allow_full_window" ;;
esac

cd experiments/referenceframe_inr_v2 || exit 1
export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"
mkdir -p outputs

echo "=== Verify_compresswin_1.3: arm=$arm mode=$mode obs=$obs frac=$frac lr=$LR wu=$WU seed=$s"
echo "    nw=$nw tau=$tau absorb=$absorb d=$db epochs=1000 n_seeds=1 ==="
python -u run_experiment.py --field "$field" --model coordnet \
    --budget_frac "$frac" --d_base "$db" --tau "$tau" \
    --absorb_min_pixels "$absorb" --n_windows "$nw" $fullwin --max_inrs 3 \
    --observer "$obs" --modes "$mode" --epochs 1000 --lr "$LR" --lr_final 1e-6 \
    --warmup_frac "$WU" --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_compresswin_1.3/${field}_${arm}_f${frac}_lr${LR}_wu${WU}_s${s}" || exit 1
echo "=== DONE task $i ==="

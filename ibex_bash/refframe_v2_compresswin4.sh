#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-21%22
#SBATCH -J RFv2win4
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_compresswin_1.4: make PROPOSED beat the mlp baseline on cylinder2d
# (user 2026-07-18: cylinder exemption is SIREN-only -- the sine basis suits
# the narrowband vortex-street oscillation; mlp has no such prior, so
# cylinder x mlp must satisfy the "not worse than baseline" criterion.
# Budget: <=24 tasks; this wave uses 22, 2 reserved.)
#
# Physics bet: the wake advects downstream at ~constant speed (Taylor frozen
# flow) -- in the co-moving frame (consttrans observer) the street becomes
# quasi-steady, exactly the transform a spectrally-biased ReLU MLP needs.
# Partition dry-run + smoke: nw=1, tau=0.1, absorb=256 -> M=1, global observer
# explains 88.6% of time energy (E/E0=0.114); byte-accounting v2 side info
# 43 B -> pro EQUAL WIDTH with bl at every tier (m=21/29/42 @ 2.5/5/10%).
# All arms mlp @ lr 3e-4 (stable everywhere, no bad basins -> no warmup),
# 1000 ep, seeds {0,7777,1}.
#
# Task layout (22):
#   i in 0..20: combo=i/3, seed={0,7777,1}[i%3]
#     combos: 0=(2.5,ct) 1=(2.5,cf) 2=(5,bl) 3=(5,ct) 4=(5,cf) 5=(10,bl) 6=(10,ct)
#   i=21: 2.5% bl seed=1 top-up (s0/s7777 exist in mainExp_compress_1.2,
#         same protocol: mlp 3e-4 1000ep)
#   5%/10% bl re-run fresh: mainExp_compress_1.1 bl was 2000ep (not comparable).
# Submit:  sbatch ibex_bash/refframe_v2_compresswin4.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

i=$SLURM_ARRAY_TASK_ID
SEEDS=(0 7777 1)

if [ "$i" -eq 21 ]; then
  arm=bl; frac=0.025; s=1
else
  combo=$((i / 3)); s=${SEEDS[$((i % 3))]}
  CARMS=( ct    cf    bl    ct    cf    bl    ct  )
  CFRAC=( 0.025 0.025 0.05  0.05  0.05  0.10  0.10 )
  arm=${CARMS[$combo]}; frac=${CFRAC[$combo]}
fi

field=cylinder2d; db=10; absorb=256; tau=0.1; LR=3e-4
case "$arm" in
  bl) mode=baseline;   obs=tvfull;     nw=2; fullwin="" ;;
  ct) mode=pro_budget; obs=consttrans; nw=1; fullwin="--allow_full_window" ;;
  cf) mode=pro_budget; obs=constfull;  nw=1; fullwin="--allow_full_window" ;;
esac

cd experiments/referenceframe_inr_v2 || exit 1
export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"
mkdir -p outputs

echo "=== Verify_compresswin_1.4: field=$field model=mlp arm=$arm mode=$mode obs=$obs"
echo "    frac=$frac nw=$nw tau=$tau absorb=$absorb d=$db lr=$LR epochs=1000 seed=$s ==="
python -u run_experiment.py --field "$field" --model mlp \
    --budget_frac "$frac" --d_base "$db" --tau "$tau" \
    --absorb_min_pixels "$absorb" --n_windows "$nw" $fullwin --max_inrs 3 \
    --observer "$obs" --modes "$mode" --epochs 1000 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_compresswin_1.4/${field}_mlp_${arm}_f${frac}_lr${LR}_s${s}" || exit 1
echo "=== DONE task $i ==="

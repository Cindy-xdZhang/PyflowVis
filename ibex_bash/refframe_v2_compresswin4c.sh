#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-11%12
#SBATCH -J RFv2win4c
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_compresswin_1.4 wave 3 (new user batch, 2026-07-19, cap <=24; this
# wave 12, remainder held for targeted follow-up): probe the mlp LR on
# cylinder at 5%. Rationale: lr was the single biggest lever of the whole
# project (6-12 dB on coordnet) yet mlp's 3e-4 came from an rfc 100-epoch
# pilot and was never validated on cylinder; existing 5% logs show MSE still
# falling 8-13% per 200ep at ep1000 under the cosine tail -> a higher lr buys
# more effective progress inside the 1000-ep budget. Symmetric arms:
#   {bl, ctM2-capsmall} x lr {5e-4, 1e-3} x 3 seeds {0,7777,1} = 12 tasks
# ctM2-capsmall = wave-2 winner structure (nw=1 tau=0.02 absorb=128 ->
# [23800,216]px, per-region consttrans, obstacle capped m=8, wake m=28).
# Submit:  sbatch ibex_bash/refframe_v2_compresswin4c.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

i=$SLURM_ARRAY_TASK_ID
SEEDS=(0 7777 1)
LRS=(5e-4 1e-3)
combo=$((i / 3)); s=${SEEDS[$((i % 3))]}
arm_idx=$((combo / 2)); LR=${LRS[$((combo % 2))]}

field=cylinder2d; db=10; frac=0.05
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

echo "=== Verify_compresswin_1.4 wave3: cylinder2d mlp arm=$arm lr=$LR f0.05 seed=$s ==="
python -u run_experiment.py --field "$field" --model mlp \
    --budget_frac "$frac" --d_base "$db" --tau "$tau" \
    --absorb_min_pixels "$absorb" --n_windows "$nw" $fullwin --max_inrs 3 \
    --observer "$obs" --alloc "$alloc" \
    --modes "$mode" --epochs 1000 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_compresswin_1.4/cylinder2d_mlp_${arm}_f${frac}_lr${LR}_s${s}" || exit 1
echo "=== DONE task $i ==="

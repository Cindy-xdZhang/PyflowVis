#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-50%51
#SBATCH -J RF3Dfix
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_rft3d_1.1 fix wave (docs 4.3.1 next-wave, user approved 2026-07-22):
# job 49256875 post-mortem found the hc160 pro collapse root cause = tvtrans
# per-frame solve pathology (frame0 |t_vec|=223 vs median 0.70 -> observed
# value range x82 -> MinMax normalization crushes the signal).  This wave:
#   (a) rerun hc160/smoke pro arms with consttrans (proct) and with the new
#       robustified tvtrans --observer_clamp 3 (protvr; clamps |t_vec| to
#       3 x median, E recomputed at the clamped q);
#   (b) lr up-probe: halfcyl bl curves were still rising at 5e-4 (2D lesson:
#       pro's relative edge grows with lr) -> {1e-3, 2e-3} for hc160/hc640;
#       smoke peaks at 3e-4 -> down-probe 2e-4.
# 17 rows x 3 seeds = 51 tasks.  All full-res, mlp, f=0.05, 1000 ep.
# Submit:  sbatch ibex_bash/refframe_3d_v1f_fixwave.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

# row = "field arm lr"; arm in {bl, proct, protvr}
ROWS=(
"halfcyl160 bl 1e-3"
"halfcyl160 bl 2e-3"
"halfcyl160 proct 5e-4"
"halfcyl160 proct 1e-3"
"halfcyl160 proct 2e-3"
"halfcyl160 protvr 5e-4"
"halfcyl160 protvr 1e-3"
"halfcyl160 protvr 2e-3"
"halfcyl640 bl 1e-3"
"halfcyl640 bl 2e-3"
"halfcyl640 proct 1e-3"
"halfcyl640 proct 2e-3"
"smoke bl 2e-4"
"smoke proct 3e-4"
"smoke proct 2e-4"
"smoke protvr 3e-4"
"smoke protvr 2e-4"
)

i=$SLURM_ARRAY_TASK_ID
SEEDS=(0 7777 1)
s=${SEEDS[$((i % 3))]}
read -r field arm LR <<< "${ROWS[$((i / 3))]}"

case "$arm" in
  bl)     mode=baseline;   obs=tvfull;     clamp=0 ;;
  proct)  mode=pro_budget; obs=consttrans; clamp=0 ;;
  protvr) mode=pro_budget; obs=tvtrans;    clamp=3 ;;
esac

cd experiments/referenceframe_inr_3d || exit 1
export PYFLOWVIS_DATA3D=${PYFLOWVIS_DATA3D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA3D=$PYFLOWVIS_DATA3D"
mkdir -p outputs

echo "=== Verify_rft3d_1.1 fixwave: $field mlp arm=$arm obs=$obs clamp=$clamp lr=$LR f0.05 seed=$s ==="
python -u run_experiment3d.py --field "$field" \
    --model mlp --budget_frac 0.05 --d_base 10 \
    --tau=-1 --n_windows 1 --max_inrs 3 \
    --observer "$obs" --observer_clamp "$clamp" \
    --modes "$mode" --epochs 1000 --batch_size 32000 \
    --max_steps_per_epoch 64 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_rft3d_1.1/${field}_mlp_${arm}_f0.05_lr${LR}_s${s}" || exit 1
echo "=== DONE task $i ==="

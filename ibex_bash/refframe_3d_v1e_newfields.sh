#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-35%36
#SBATCH -J RF3Dnew
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_rft3d_1.1 part-3: three new full-resolution 3D fields, chosen from the
# local flowData3D folder by dry-run E/E0 (job of 2026-07-21):
#   halfcyl160  E/E0=0.137 (tv gap real -> observer tvtrans)   ~ 2D cylinder's 3D kin
#   halfcyl640  E/E0=0.167 (variants flat -> consttrans)       same geometry, 4x Re
#   smoke       E/E0=0.515 (tv gap real -> tvtrans; upward t_y~1.4, bouss 3D kin)
# Together with deltaWing (0.70) this spans the criterion axis; per-field
# expected gain ordering hc160 >= hc640 >> deltawing is ITSELF the prediction
# under test.  All fields run FULL RES (they are 6-15x smaller than deltaWing).
#   {hc160, hc640, smoke} x {bl, proM1} x lr {5e-4, 3e-4} x 3 seeds = 36 tasks
# Submit:  sbatch ibex_bash/refframe_3d_v1e_newfields.sh

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
field_idx=$((combo / 4))
cb=$((combo % 4))
arm_idx=$((cb / 2))
LRS=(5e-4 3e-4); LR=${LRS[$((cb % 2))]}

FIELDS=(halfcyl160 halfcyl640 smoke)
OBSERVERS=(tvtrans consttrans tvtrans)   # per-field, from dry-run E/E0
field=${FIELDS[$field_idx]}
obs=${OBSERVERS[$field_idx]}

if [ "$arm_idx" == "0" ]; then
  arm=bl; mode=baseline
else
  arm=pro; mode=pro_budget
fi

cd experiments/referenceframe_inr_3d || exit 1
export PYFLOWVIS_DATA3D=${PYFLOWVIS_DATA3D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA3D=$PYFLOWVIS_DATA3D"
mkdir -p outputs

echo "=== Verify_rft3d_1.1 newfields: $field mlp arm=$arm obs=$obs lr=$LR f0.05 seed=$s ==="
python -u run_experiment3d.py --field "$field" \
    --model mlp --budget_frac 0.05 --d_base 10 \
    --tau=-1 --n_windows 1 --max_inrs 3 --observer "$obs" \
    --modes "$mode" --epochs 1000 --batch_size 32000 \
    --max_steps_per_epoch 64 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_rft3d_1.1/${field}_mlp_${arm}_f0.05_lr${LR}_s${s}" || exit 1
echo "=== DONE task $i ==="

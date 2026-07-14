#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-19
#SBATCH -J RFv2tau
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_tau_1.1: tau sensitivity of the proposed method (pro_quality vs baseline).
# One (dataset, tau, seed) per array task -- 20 parallel tasks:
#   tasks 0-9  : cylinder2d, tau in {0.005 0.01 0.02 0.05 0.1}, absorb=64,
#                seeds {0, 7777}   (dry-run M = 21/14/4/5/3)
#   tasks 10-19: boussinesq, tau in {0.1 0.15 0.2 0.3 0.5}, absorb=256,
#                seeds {0, 7777}   (dry-run M = 24/21/15/7/2)
# Baselines are NOT rerun here (tau-independent): reuse Verify_seedstability_1.1
# (cylinder, seeds 10-17) and the boussinesq mainExp baseline tasks.
# Submit:  sbatch ibex_bash/refframe_v2_tausweep.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"

i=$SLURM_ARRAY_TASK_ID
cd experiments/referenceframe_inr_v2 || exit 1
mkdir -p outputs

if [ "$i" -lt 10 ]; then
  FIELD=cylinder2d; ABSORB=64
  TAUS=(0.005 0.01 0.02 0.05 0.1)
  j=$i
else
  FIELD=boussinesq; ABSORB=256
  TAUS=(0.1 0.15 0.2 0.3 0.5)
  j=$((i - 10))
fi
tau=${TAUS[$((j / 2))]}
if [ $((j % 2)) -eq 0 ]; then s=0; else s=7777; fi

echo "=== Verify_tau_1.1: $FIELD tau=$tau absorb=$ABSORB seed=$s ==="
python -u run_experiment.py --field "$FIELD" --m_base 64 --d_base 10 \
    --tau "$tau" --absorb_min_pixels "$ABSORB" --modes pro_quality \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/ibex_tausweep/${FIELD}_tau${tau}_s${s}" || exit 1
echo "=== DONE task $i ==="

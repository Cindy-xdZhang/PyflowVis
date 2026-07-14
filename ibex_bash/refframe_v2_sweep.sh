#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-15
#SBATCH -J RFv2sw
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Parallel sweep: one (field, mode, seed) per array task -- max parallelism on Ibex.
#   tasks 0-7  : boussinesq mainExp_2.3-ibex, 4 modes x 2 seeds (seeds 0 / 7777 match
#                the in-process best-of-2 pair; selection done post-hoc, run-level)
#   tasks 8-15 : cylinder2d baseline seeds 10..17 (Verify_seedstability_1.1-ibex --
#                distribution study of the m=64 SIREN convergence fork)
# Submit:  sbatch ibex_bash/refframe_v2_sweep.sh

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

if [ "$i" -lt 8 ]; then
  MODES=(baseline pro_budget pro_quality no_observer)
  mode=${MODES[$((i / 2))]}
  if [ $((i % 2)) -eq 0 ]; then s=0; else s=7777; fi
  echo "=== boussinesq mode=$mode seed=$s ==="
  python -u run_experiment.py --field boussinesq --m_base 64 --d_base 10 \
      --tau 0.2 --absorb_min_pixels 256 --modes "$mode" --seed "$s" --n_seeds 1 \
      --out_dir "outputs/ibex_v23_bouss/${mode}_s${s}" || exit 1
else
  s=$((i - 8 + 10))
  echo "=== cylinder2d baseline seed=$s ==="
  python -u run_experiment.py --field cylinder2d --m_base 64 --d_base 10 \
      --tau 0.1 --modes baseline --seed "$s" --n_seeds 1 \
      --out_dir "outputs/ibex_v23_cylbase/s${s}" || exit 1
fi
echo "=== DONE task $i ==="

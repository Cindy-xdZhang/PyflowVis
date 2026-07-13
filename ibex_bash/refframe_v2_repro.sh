#!/bin/bash
#SBATCH -N 1
#SBATCH -J RFv2
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Reproduce the referenceframe_inr_v2 v2.3 experiments on Ibex (cross-machine check
# of docs/referenceframe_inr_v2.md par.4.4b conclusions). Usage:
#   sbatch ibex_bash/refframe_v2_repro.sh rfc        # validate suite + rfc 4 modes (best-of-3)
#   sbatch ibex_bash/refframe_v2_repro.sh rfc_diag   # rfc single-window attribution diagnostic
#   sbatch ibex_bash/refframe_v2_repro.sh cylinder   # cylinder2d 4 modes (best-of-2)
# Data: set PYFLOWVIS_DATA2D below to the Ibex folder holding cylinder2d.nc.

[ ! -d "slurm_logs" ] && echo "Create a directory slurm_logs" && mkdir -p slurm_logs

module load cuda/11.8
echo "===> load cuda/11.8"
source ~/.bashrc
conda activate deepvortex

nvidia-smi
hostname

export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/flowData2D}
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"

TARGET=${1:-rfc}
cd experiments/referenceframe_inr_v2 || exit 1
mkdir -p outputs

case $TARGET in
  rfc)
    echo "=== validate_rfc ==="
    python -u validate_rfc.py || exit 1
    echo "=== rfc v2.3 (4 modes, best-of-3) ==="
    python -u run_experiment.py --field rfc --n_seeds 3 --out_dir outputs/ibex_v23
    ;;
  rfc_diag)
    echo "=== rfc v2.3 single-window diagnostic ==="
    python -u run_experiment.py --field rfc --n_windows 1 --allow_full_window \
        --modes pro_budget,no_observer --n_seeds 3 --out_dir outputs/ibex_v23_diag1w
    ;;
  cylinder)
    echo "=== cylinder2d v2.3 (4 modes, best-of-2) ==="
    python -u run_experiment.py --field cylinder2d --m_base 64 --d_base 10 \
        --tau 0.1 --n_seeds 2 --out_dir outputs/ibex_v23
    ;;
  *)
    echo "unknown target '$TARGET'"; exit 1
    ;;
esac
echo "=== DONE $TARGET ==="

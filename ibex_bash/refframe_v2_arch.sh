#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-39%32
#SBATCH -J RFv2arch
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_arch_1.1: INR architecture variants v_MLP0.0 (residual ReLU) and
# v_FINER0.0 (FINER, variable-periodic sine) on 5 fields, baseline vs
# pro_quality WITHIN each architecture (does the observer transform help when
# the INR is not SIREN?).
#
# Protocol changes (user 2026-07-15):
#   * epochs hard cap relaxed 1000 -> 2000; lr chosen PER ARCHITECTURE by a
#     local pilot (rfc 200ep): mlp 3e-4 > 1e-4; finer 1e-4 >> official 5e-4
#     (5e-4 collapses even on rfc m=24,d=4: MSE parked at 1.087e-1).
#     Converged quality is what matters; papers report model size only.
#   * 2 independent seeds {0, 7777}, conclusions = MEAN of the two runs
#     (replaces best-of-k; n_seeds=1 per task, no encode-time seed search).
#
# One (field, model, mode, seed) per array task -> 40 tasks, <=32 concurrent:
#   i = 8*field_idx + 4*model_idx + 2*mode_idx + seed_idx
# Field configs (tau/absorb = best recorded config per field):
#   rfc        m=24 d=4   tau=0.05 absorb=0    (mainExp_2.3, N=1 per window)
#   cylinder2d m=64 d=10  tau=0.1  absorb=0    (mainExp_2.3 pro_quality 62.4-62.8, M=5)
#   boussinesq m=64 d=10  tau=0.5  absorb=256  (Verify_tau_1.1 win 70.43, M=2)
#   gerris0    m=64 d=10  tau=0.6  absorb=0    (Verify_gerristiny_1.1 pilot, 9 INRs)
#   gerris4    m=64 d=10  tau=0.7  absorb=0    (pilot tau=0.6 TIMEOUT -> 0.7, N~11)
# Data: gerris* under /ibex/user/zhanx0o/FLowDataFolder; cylinder/boussinesq
# under $HOME/DeepVortex/FLowDataFolder (rfc is synthetic, no data needed).
# Submit:  sbatch ibex_bash/refframe_v2_arch.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

i=$SLURM_ARRAY_TASK_ID

FIELDS=(rfc cylinder2d boussinesq gerris0 gerris4)
MB=(24 64 64 64 64)
DB=(4 10 10 10 10)
TAUS=(0.05 0.1 0.5 0.6 0.7)
ABSORB=(0 0 256 0 0)
MODELS=(mlp finer)
MODES=(baseline pro_quality)
SEEDS=(0 7777)

f=$((i / 8)); j=$((i % 8))
field=${FIELDS[$f]}; mb=${MB[$f]}; db=${DB[$f]}; tau=${TAUS[$f]}; absorb=${ABSORB[$f]}
model=${MODELS[$((j / 4))]}
mode=${MODES[$(((j / 2) % 2))]}
s=${SEEDS[$((j % 2))]}

if [ "$model" == "mlp" ]; then LR=3e-4; else LR=1e-4; fi

if [[ "$field" == gerris* ]]; then
  export PYFLOWVIS_DATA2D=/ibex/user/zhanx0o/FLowDataFolder
else
  export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/FLowDataFolder}
fi
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"

cd experiments/referenceframe_inr_v2 || exit 1
mkdir -p outputs

echo "=== Verify_arch_1.1: field=$field model=$model mode=$mode seed=$s"
echo "    m=$mb d=$db tau=$tau absorb=$absorb lr=$LR epochs=2000 n_seeds=1 ==="
python -u run_experiment.py --field "$field" --model "$model" \
    --m_base "$mb" --d_base "$db" --tau "$tau" --absorb_min_pixels "$absorb" \
    --modes "$mode" --epochs 2000 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_arch_1.1/${field}_${model}_${mode}_s${s}" || exit 1
echo "=== DONE task $i ==="

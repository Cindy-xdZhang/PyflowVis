#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-3
#SBATCH -J RFgerris
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_gerristiny_1.1 pilot: gerris0 (small |v|~0.066) + gerris4 (large |v|~0.86),
# baseline vs pro_quality, recipe v2.3 (lr 1e-5 cosine, <=1000 epochs, best-of-3 seeds,
# adaptive batch). GerrisTinySet is downsampled to (T,X,Y)=(128,128,256) inside load_field.
# One (field, mode) per array task -> 4 parallel tasks.
#   task 0: gerris0 baseline
#   task 1: gerris0 pro_quality
#   task 2: gerris4 baseline
#   task 3: gerris4 pro_quality
# Submit:  sbatch ibex_bash/refframe_gerris_pilot.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

export PYFLOWVIS_DATA2D=/ibex/user/zhanx0o/FLowDataFolder
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"

i=$SLURM_ARRAY_TASK_ID
cd experiments/referenceframe_inr_v2 || exit 1
mkdir -p outputs

FIELDS=(gerris0 gerris0 gerris4 gerris4)
MODES=(baseline pro_quality baseline pro_quality)
field=${FIELDS[$i]}
mode=${MODES[$i]}

echo "=== Verify_gerristiny_1.1: field=$field mode=$mode (recipe v2.3) ==="
python -u run_experiment.py --field "$field" --m_base 64 --d_base 10 \
    --tau 0.6 --modes "$mode" --n_seeds 3 \
    --out_dir "outputs/Verify_gerristiny_1.1/${field}_${mode}" || exit 1
echo "=== DONE task $i ==="

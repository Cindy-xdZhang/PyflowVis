#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-71%32
#SBATCH -J RFv2cmp
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# mainExp_compress_1.1: strict-compression protocol -- the INR total bytes
# (params*4 + side info) must fit a FRACTION of the raw float32 field bytes
# (5% / 10% / 20%, i.e. CR >= 20x / 10x / 5x), stricter than the old
# pro_budget (B was ~19-23% of the field). Per architecture (coordnet = the
# Coordinate INR baseline, mlp = v_MLP0.0 residual ReLU), compare
#   baseline    one INR fits v directly, width solved from the byte budget
#   pro_budget  tau-merge partition + per-region killing observer + observed-
#               field INRs; side info deducted from the SAME byte budget
# Sizing preview: experiments/referenceframe_inr_v2/budget_calc.py
#   rfc        d=4  M=2: bl m=12/17/24, pro m_r=8/12/17   @5/10/20%
#   cylinder2d d=10 M=5: bl m=29/42/60, pro m_r=13/18/26
#   boussinesq d=10 M=2: bl m=24/34/48, pro m_r=16/24/34
#
# Protocol = Verify_arch_1.1 (user 2026-07-15): epochs 2000, per-arch lr
# (coordnet 1e-5 = frozen v2.3 recipe; mlp 3e-4 from lr pilot), 2 independent
# seeds {0, 7777} as separate tasks, conclusions = MEAN of the two.
#
# One (field, frac, model, mode, seed) per array task -> 72 tasks, <=32 conc.:
#   i = 24*field_idx + 8*frac_idx + 4*model_idx + 2*mode_idx + seed_idx
# Field configs (tau/absorb = recorded operating point per field, docs par.4.4j):
#   rfc        d=4   tau=0.05 absorb=0    (mainExp_2.3, N=1 per window)
#   cylinder2d d=10  tau=0.1  absorb=0    (mainExp_2.3, M=5)
#   boussinesq d=10  tau=0.5  absorb=256  (Verify_tau_1.1 win point, M=2)
# Data: cylinder/boussinesq under $HOME/DeepVortex/FLowDataFolder (rfc is
# synthetic, no data needed).
# Submit:  sbatch ibex_bash/refframe_v2_compress.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

i=$SLURM_ARRAY_TASK_ID

FIELDS=(rfc cylinder2d boussinesq)
DB=(4 10 10)
TAUS=(0.05 0.1 0.5)
ABSORB=(0 0 256)
FRACS=(0.05 0.10 0.20)
MODELS=(coordnet mlp)
MODES=(baseline pro_budget)
SEEDS=(0 7777)

f=$((i / 24)); r=$((i % 24))
field=${FIELDS[$f]}; db=${DB[$f]}; tau=${TAUS[$f]}; absorb=${ABSORB[$f]}
frac=${FRACS[$((r / 8))]}
j=$((r % 8))
model=${MODELS[$((j / 4))]}
mode=${MODES[$(((j / 2) % 2))]}
s=${SEEDS[$((j % 2))]}

if [ "$model" == "mlp" ]; then LR=3e-4; else LR=1e-5; fi
# Optional lr override for reruns (pattern from refframe_v2_arch.sh): submit
#   sbatch --export=ALL,RFV2_LR=<lr> --array=<indices> <this script>
# Overridden runs write to a *_lr<LR> suffixed out_dir, never clobbering.
OUTSUF=""
if [ -n "$RFV2_LR" ]; then LR=$RFV2_LR; OUTSUF="_lr${LR}"; fi

export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"

cd experiments/referenceframe_inr_v2 || exit 1
mkdir -p outputs

echo "=== mainExp_compress_1.1: field=$field frac=$frac model=$model mode=$mode seed=$s"
echo "    d=$db tau=$tau absorb=$absorb lr=$LR epochs=2000 n_seeds=1 ==="
python -u run_experiment.py --field "$field" --model "$model" \
    --budget_frac "$frac" --d_base "$db" --tau "$tau" \
    --absorb_min_pixels "$absorb" \
    --modes "$mode" --epochs 2000 --lr "$LR" --lr_final 1e-6 \
    --seed "$s" --n_seeds 1 \
    --out_dir "outputs/mainExp_compress_1.1/${field}_${model}_${mode}_f${frac}_s${s}${OUTSUF}" || exit 1
echo "=== DONE task $i ==="

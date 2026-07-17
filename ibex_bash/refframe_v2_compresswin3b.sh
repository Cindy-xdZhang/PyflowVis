#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-14%15
#SBATCH -J RFv2win3b
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[a100|v100]
#SBATCH --mem=64G

# Verify_compresswin_1.3 wave 2 (docs par.4.4o): close the boussinesq 5% cell.
# Wave-1 verdicts: 2.5% WON (cf@1e-4+wu 64.48 vs bl best 63.73, three pro arms
# above bl optimum); 5% still -1.09 (ct@7e-5+wu 65.99 vs bl@1e-4+wu 67.08).
# Wave-2 bets, from wave-1 attribution:
#   * cf (constfull) was the strongest 2.5% arm (+0.45 over ct) but was never
#     run at 5% -> cf x {7e-5, 1e-4} x wu0.1.
#   * ct@1e-4+wu0.1 has a residual left tail (s1=58.07) -> longer warmup
#     wu0.2 for {cf, ct}@1e-4, with bl@1e-4+wu0.2 run symmetrically (fairness;
#     bl@1e-4+wu0.1 is already stable so expected ~neutral for bl).
# 5 combos x 3 seeds {0, 7777, 1} = 15 tasks. Total new-deployment budget this
# session: 45 (wave 1, job 49044332) + 15 = 60 <= 64 (user cap).
# Submit:  sbatch ibex_bash/refframe_v2_compresswin3b.sh

[ ! -d "slurm_logs" ] && mkdir -p slurm_logs
module load cuda/11.8
source ~/.bashrc
conda activate deepvortex
nvidia-smi --query-gpu=name --format=csv,noheader
hostname

#        0      1      2      3      4
ARMS=(  cf     cf     cf     ct     bl  )
LRS=(   7e-5   1e-4   1e-4   1e-4   1e-4)
WUS=(   0.1    0.1    0.2    0.2    0.2 )
SEEDS=( 0 7777 1 )

i=$SLURM_ARRAY_TASK_ID
combo=$((i / 3))
s=${SEEDS[$((i % 3))]}
arm=${ARMS[$combo]}
LR=${LRS[$combo]}
WU=${WUS[$combo]}
frac=0.05

field=boussinesq; db=10; absorb=256; tau=0.5
case "$arm" in
  bl) mode=baseline;   obs=tvfull;     nw=2; fullwin="" ;;
  ct) mode=pro_budget; obs=consttrans; nw=1; fullwin="--allow_full_window" ;;
  cf) mode=pro_budget; obs=constfull;  nw=1; fullwin="--allow_full_window" ;;
esac

cd experiments/referenceframe_inr_v2 || exit 1
export PYFLOWVIS_DATA2D=${PYFLOWVIS_DATA2D:-$HOME/DeepVortex/FLowDataFolder}
echo "PYFLOWVIS_DATA2D=$PYFLOWVIS_DATA2D"
mkdir -p outputs

echo "=== Verify_compresswin_1.3 wave2: arm=$arm mode=$mode obs=$obs frac=$frac lr=$LR wu=$WU seed=$s"
echo "    nw=$nw tau=$tau absorb=$absorb d=$db epochs=1000 n_seeds=1 ==="
python -u run_experiment.py --field "$field" --model coordnet \
    --budget_frac "$frac" --d_base "$db" --tau "$tau" \
    --absorb_min_pixels "$absorb" --n_windows "$nw" $fullwin --max_inrs 3 \
    --observer "$obs" --modes "$mode" --epochs 1000 --lr "$LR" --lr_final 1e-6 \
    --warmup_frac "$WU" --seed "$s" --n_seeds 1 \
    --out_dir "outputs/Verify_compresswin_1.3/${field}_${arm}_f${frac}_lr${LR}_wu${WU}_s${s}" || exit 1
echo "=== DONE task $i ==="

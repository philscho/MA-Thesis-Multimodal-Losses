#! /bin/bash
#SBATCH --job-name=test
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --no-requeue
#SBATCH --time=72:00:00
# activate conda env
eval "$(conda shell.bash hook)"
conda activate /scratch/vihps/vihps19/py3.9
#debugging flags
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
srun python3 lightning_test.py

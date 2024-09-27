#! /bin/bash
#SBATCH --job-name=coco
#SBATCH --output=/home/vihps/vihps19/multimodal/slurm/logs/%j.out
#SBATCH --error=/home/vihps/vihps19/multimodal/slurm/logs/%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --no-requeue
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:8
# activate conda env
# eval "$(conda shell.bash hook)"
source activate /scratch/vihps/vihps19/multimodal
#debugging flags
#export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "---------------------------------------------"
if [ "$1" == "overfit" ]; then
    CONFIG="/home/vihps/vihps19/multimodal/configs/config_HLR_overfit.yaml"
else
    CONFIG="/home/vihps/vihps19/multimodal/configs/config_HLR.yaml"
fi
echo "Running: srun python3 train_model_coco_dualenc_new.py --config $CONFIG $@"
srun python3 /home/vihps/vihps19/multimodal/train_model_coco_dualenc_new.py --config $CONFIG $@
echo "---------------------------------------------"

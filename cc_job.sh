#!/bin/bash
#SBATCH --mail-user=07nino@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=PPOimgIris
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=190GB
#SBATCH --time=1-03:30
#SBATCH --account=rrg-ebrahimi

source ~/scratch/envs/iris3/bin/activate

xvfb-run -a python3 src/main.py env.train.id=mpe.simple_spread common.device=cuda:0 wandb.mode=offline


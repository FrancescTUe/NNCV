#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh
#srun apptainer exec --nv --env-file .env container.sif /bin/bash main_entropy.sh
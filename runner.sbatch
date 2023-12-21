#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=flir-trainer
#SBATCH --mail-type=ALL
#SBATCH --partition=cuda
#SBATCH --mem-per-cpu=160G
#SBATCH --time=20:00:00

module load nvidia/cudasdk/11.6
module load intel/python/3

source .venv/bin/activate

python trainer.py

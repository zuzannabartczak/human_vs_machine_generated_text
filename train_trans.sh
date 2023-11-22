#!/bin/bash
#SBATCH -A uppmax2023-2-19
#SBATCH -p core -n 4
#SBATCH -M snowy
#SBATCH -t 00:15:00
#SBATCH -J transmodelA
#SBATCH --gres=gpu:1
#SBATCH --qos=short

module load python
 
python /domus/h1/zuzanna/hvm/transmodela.py
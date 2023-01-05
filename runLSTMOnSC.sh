#!/bin/bash
#SBATCH -J ModelTrainerSortedByTimeLSTM
#SBATCH --account=def-arunita
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G # 512GiB of memery
#SBATCH -t 0-08:00 # Running time of 10 min is 0-00:10
. ~/ENV/bin/activate
python ModelTrainerSCSortedByTimeLSTM.py

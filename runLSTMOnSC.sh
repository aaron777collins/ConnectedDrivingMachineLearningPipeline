#!/bin/bash
#SBATCH -J ModelTrainerSortedByTimeLSTM
#SBATCH --account=def-arunita
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G # 256GiB of memery
#SBATCH -t 0-02:0 # Running time of 10 min is 0-00:10
. ~/ENV/bin/activate
python ModelTrainerSCSortedByTimeLSTM.py

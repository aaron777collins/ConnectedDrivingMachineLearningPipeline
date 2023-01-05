#!/bin/bash
#SBATCH -J UploadToGit
#SBATCH --account=def-arunita
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G # 64GiB of memery
#SBATCH -t 0-00:30 # Running time of 10 min is 0-00:10
git add .
git commit -m "update"
git push

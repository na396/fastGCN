#!/bin/bash -l
#SBATCH --job-name=fastGCN_Script2         # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=na396@njit.edu    # Where to send mail	
#SBATCH --ntasks=16                   # Run on a single CPU
#SBATCH --mem=0                       # Job memory request
#SBATCH -p datasci                    # Partition
#SBATCH  --gres=gpu:1                 # gpu
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module purge  > /dev/null 2>&1
conda activate Py-env
srun python main.py
#!/bin/bash
#SBATCH --partition=dggpu

# Request Nodes
#SBATCH --nodes=1

# Req CPU Cores
#SBATCH --ntasks=2

# Req GPUs
#SBATCH --gres=gpu:4

# Req Memory
#SBATCH --mem=50G

# Run for 20 minutes
#SBATCH --time=120

# Name of job
#SBATCH --job-name=TestClay254Final

# Name output file
#SBATCH --output=%x_j%j.out

# Email
#SBATCH --mail-type=ALL

source ~/.bashrc

cd ${SLURM_SUBMIT_DIR}
# Executable section: echoing some Slurm data
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "Job ID:          ${SLURM_JOBID}"
conda activate cs254
time python clay_vacc.py


#!/bin/bash
#SBATCH -p normal
#SBATCH -J pymoo_imc
#SBATCH -o pymoo_imc.out
#SBATCH -e pymoo_imc.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=50
#SBATCH -t 24:00:00

export PYTHONPATH=$PYTHONPATH:~/repos/neuron_fitting
uv run adexlif_ot_ficurve/scripts/adexlif_pymoo_nsga.py --neuron_type imc --n-gen 400 --pop-size 200 --n-workers 50 --algorithm NSGA2 --seed 0 


#!/bin/bash
#BATCH --time=1000:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --partition=cpulong
#SBATCH --job-name=adjoint_generic_ms_im
#SBATCH --err=adjoint_generic_ms_im.err
#SBATCH --out=adjoint_generic_ms_im.out

ml tqdm/4.66.2-GCCcore-13.2.0
ml PyTorch/2.3.0-foss-2023b
ml matplotlib/3.8.2-gfbf-2023b
ml h5py/3.11.0-foss-2023b
ml Optuna/3.6.1-foss-2023b

python run_generic_adjoint.py --dynamics_name "double_well" --data_folder "double_well_20" --gt_res 0.01 --hid_layers "[16, 32, 16]" --solver_res 0.01 --pred True --sim_len 7 --solver "im"
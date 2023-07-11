### SNAP grid performance

This directory contains benchmarks on a single Perlmutter node.
Benchmarks were obtained with the following calculations.
The `test_snap.py` script loops over a num. grid point list for `nbox` atoms and save a `metrics_{nbox}_kk{kkflag}.dat` file.

For running on 4 GPUs:
    
    srun --ntasks-per-node=4 --gpus-per-node=4 -c 32 python test_snap.py 1 3
    
For running on CPU with 64 cores:
    
    srun -n 64 --cpu-bind=cores -c 2 python test_snap.py 1 3
    
Do this for various `nbox` args (the second argument), to get various metric files with number of atoms. 

### SNAP grid performance for strong scaling

Here we keep system size constant (number of grid points and number of atoms) and observe scaling with nodes.

For these examples, `nbox = 7` which is `4*7**3 = 1372` atoms.

For a single GPU, simple do:

    python test_snap_1gpu.py 1 7 1

For running on 2 nodes, 4 GPUs each:
    
    srun --ntasks-per-node=4 --gpus-per-node=4 -c 32 -N 2 python test_snap.py 1 7 2

See

`run_1nodes.sh`
`run_2nodes.sh`
`run_10nodes.sh`
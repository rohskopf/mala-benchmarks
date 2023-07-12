"""
Script for Kokkos grid benchmarks

Usage:
    python test.py kkflag nbox

Args:
    kkflag: 0 if not using kokkos, 1 if using kokkos
    nbox: number of unit cell replicas, total box is nbox*nbox*nbox

"""

import lammps as lammps
from lammps import constants as lammps_constants
import numpy as np
import ctypes
import sys
import time
import matplotlib.pyplot as plt
from mpi4py import MPI

def extract_compute_np_fs(lmp, name, compute_style, result_type,
                          array_shape=None):
    """
    This function is used in FitSNAP.
    Convert a lammps compute to a numpy array.
    Assumes the compute stores floating point numbers.
    Note that the result is a view into the original memory.
    If the result type is 0 (scalar) then conversion to numpy is
    skipped and a python float is returned.
    From LAMMPS/src/library.cpp:
    style = 0 for global data, 1 for per-atom data, 2 for local data
    type = 0 for scalar, 1 for vector, 2 for array
    """

    if array_shape is None:
        array_np = lmp.numpy.extract_compute(name, compute_style, result_type)

    else:
        ptr = lmp.extract_compute(name, compute_style, result_type)
        if result_type == 0:
            # no casting needed, lammps.py already works
            return ptr

        if result_type == 2:
            ptr = ptr.contents

        total_size = np.prod(array_shape)
        buffer_ptr = ctypes.cast(ptr,
                                 ctypes.POINTER(ctypes.c_double * total_size))

        array_np = np.frombuffer(buffer_ptr.contents, dtype=float)
        array_np.shape = array_shape

    return array_np

def calculate(lmp, kkflag, ngrid, nbox, rcutfac, twojmax, switch, radelem, sigma): 
    
    rcutneigh = 2.0*rcutfac*radelem
    
    rfac0 = 0.99363
    rmin0 = 0
    wj = 1
    radelem = 0.5
    bzero = 0
    quadratic = 0
    switch = 1

    cmd_str = \
    f"""
    # pass in values ngridx, ngridy, ngridz, twojmax, rcutfac, atom_config_fname 
    # using command-line -var option

    # Initialize simulation

    units     metal

    boundary        p p p

    # beryllium
    #lattice         hcp 2.267
    #region          box block 0 {nbox} 0 {nbox} 0 {nbox}
    #create_box      1 box
    #create_atoms    1 box
    #mass 1 9.0

    # aluminum
    lattice         fcc 4.05
    region          box block 0 {nbox} 0 {nbox} 0 {nbox}
    create_box      1 box
    create_atoms    1 box
    mass 1 26.98

    # Needs to be defined for Kokkos
    run_style verlet

    # define grid compute and atom compute

    group     snapgroup type 1
    variable   rfac0 equal 0.99363
    variable   rmin0 equal 0
    variable   wj equal 1
    #variable   radelem equal 0.5
    #variable   bzero equal 0
    #variable   quadratic equal 0
    #variable   sigma equal 1.0
    
    compute bgrid all sna/grid/local grid {ngrid} {ngrid} {ngrid} {rcutfac} {rfac0} {twojmax} {radelem} {wj} rmin0 {rmin0} bzeroflag {bzero} quadraticflag {quadratic} switchflag {switch}

    pair_style zero {rcutneigh}
    pair_coeff * *

    # define output

    thermo_style   custom step temp ke pe vol
    thermo_modify norm yes

    run 0
    """

    lmp.commands_string(cmd_str)
    
    natoms = lmp.get_natoms()

    time1 = time.perf_counter()
    test_grid_arr = extract_compute_np_fs(lmp, "bgrid", lammps_constants.LMP_STYLE_LOCAL, 2)
    time2 = time.perf_counter()
    comptime = time2 - time1
    #print(f"Comptime shape: {comptime} {np.shape(test_grid_arr)}")
    
    # Uncomment if want to save the local arrays:
    """
    if kkflag:
        np.save(f"test_kokkos_{rank}.npy", test_grid_arr)
    else:
        np.save(f"test_cpu_{rank}.npy", test_grid_arr)
    """
    
    lmp.command("clear")

    return comptime


# Declare settings.
# Declare ngrid settings to loop over.
# nbox_lst = [1,2,4,8,10] 
# 10M, 100M
ngrid_lst = [220] #[5, 10, 20, 50, 70, 90, 100, 300, 500]

kkflag = kkflag = int(sys.argv[1])
nbox = int(sys.argv[2])
nnodes = int(sys.argv[3])
#print(f"{kkflag} {nbox}")

rcutfac = 5.0
twojmax = 8
switch = 0
radelem = 0.5
sigma = 1.0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if (kkflag):
    print(f"Using kokkos on rank {rank}")
    lmp = lammps.lammps(comm=comm,
                        cmdargs=["-k", "on", "g", str(size), "-sf", "kk", 
                                 "-v", "atom_config_fname", "Be16_cubic_input.tmp",
                                 "-v", "switch", "0"])
else:
    print(f"Not using kokkos on rank {rank}")
    lmp = lammps.lammps(comm=comm,
                        cmdargs=["-v", "atom_config_fname", "Be16_cubic_input.tmp",
                                 "-v", "switch", "0"])

#for nbox in nbox_lst[0:2]:
metric_lst = []
npoint_lst = []
for ngrid in ngrid_lst:
    comptime_p = calculate(lmp, kkflag, ngrid, nbox, rcutfac, twojmax, switch, radelem, sigma)
    #comptime_p = np.array([comptime_p])
    #comptime_reduced = np.array([0.0])
    #comm.Allreduce([comptime_p, MPI.DOUBLE], [comptime_reduced, MPI.DOUBLE])
    comptime = comptime_p #comptime_reduced[0]/size
    #print(f">>> Avg. comptime: {comptime}")
    numpoints = (ngrid**3)/1e6
    npoint_lst.append(numpoints)
    #metric_lst.append(numpoints/comptime)
    metric_lst.append(comptime)
    #comm.Barrier()
npoints = np.array([npoint_lst]).T
metrics = np.array([metric_lst]).T
# Divide metrics by number of nodes
#metrics = metrics/nnodes
print(npoints)
print(metrics)
dat = np.concatenate((npoints,metrics), axis=1)
if rank == 0:
    print(f">>> saving metrics on rank 0, size {size}")
    np.savetxt(f"metrics_{nbox}_kk{kkflag}_{size}gpus.dat", dat)

# Need to finalize else get CUDA finalize error.
lmp.finalize()
lmp.close()
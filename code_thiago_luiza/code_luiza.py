####################################################################
# Import libraries
####################################################################
import numpy as np
from mpi4py import MPI

from tools import input_simulation_parameters
from tools import DirichletBC3D
from simuladores.slab import slab3D
from tools import run_advection_solver_3D
from tools import InitialConditionOilStain
####################################################################

####################################################################
# Inicializacao MPI
####################################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  
####################################################################

internal_simulpar = input_simulation_parameters('simulation_input2.in')

# Dimensões globais
Gnx, Gny, Gnz = internal_simulpar.mesh
GLx, GLy, GLz = internal_simulpar.Dom

# Cada rank vai pegar uma parte de X
nx_local = Gnx // size

####################################################################
# Loop que percorre os 4 processos
####################################################################
i = 0
for _ in range(size):

    if rank == i:
        print(f"[Rank {rank}] executando...")

        nx = nx_local
        ny = Gny
        nz = Gnz
        Lx = GLx
        Ly = GLy 
        Lz = GLz
     
        Y = 10 + 0 * np.random.rand(nx, ny, nz)
        internal_simulpar.BC = DirichletBC3D

        coord, K, psim, dsim, vxsim, vysim, vzsim, divsim = slab3D(internal_simulpar, Y)

        cfl = 1.0
        day = 86400
        tf = 60 * day
        IC = InitialConditionOilStain(20, 80, 20, 80, 0, nz, Lx/nx, Ly/ny, Lz/nz)

        x, y, z, c_hist, dt, nt = run_advection_solver_3D(
            Lx, Ly, Lz, nx, ny, nz,
            ux=vxsim, uy=vysim, uz=vzsim,
            cfl=cfl, tf=tf, IC=IC
        )
        i += 1

    if rank == i + 1:
        print(f"[Rank {rank}] executando...")

        nx = nx_local
        ny = Gny
        nz = Gnz
        Lx = GLx
        Ly = GLy 
        Lz = GLz

        Y = 10 + 0 * np.random.rand(nx, ny, nz)
        internal_simulpar.BC = DirichletBC3D

        coord, K, psim, dsim, vxsim, vysim, vzsim, divsim = slab3D(internal_simulpar, Y)

        cfl = 1.0
        day = 86400
        tf = 60 * day
        IC = InitialConditionOilStain(20, 80, 20, 80, 0, nz, Lx/nx, Ly/ny, Lz/nz)

        x, y, z, c_hist, dt, nt = run_advection_solver_3D(
            Lx, Ly, Lz, nx, ny, nz,
            ux=vxsim, uy=vysim, uz=vzsim,
            cfl=cfl, tf=tf, IC=IC
        )
        i += 1

    if rank == i + 2:
        print(f"[Rank {rank}] executando...")

        nx = nx_local
        ny = Gny
        nz = Gnz
        Lx = GLx
        Ly = GLy 
        Lz = GLz

        Y = 10 + 0 * np.random.rand(nx, ny, nz)
        internal_simulpar.BC = DirichletBC3D

        coord, K, psim, dsim, vxsim, vysim, vzsim, divsim = slab3D(internal_simulpar, Y)

        cfl = 1.0
        day = 86400
        tf = 60 * day
        IC = InitialConditionOilStain(20, 80, 20, 80, 0, nz, Lx/nx, Ly/ny, Lz/nz)

        x, y, z, c_hist, dt, nt = run_advection_solver_3D(
            Lx, Ly, Lz, nx, ny, nz,
            ux=vxsim, uy=vysim, uz=vzsim,
            cfl=cfl, tf=tf, IC=IC
        )
        i += 1

    if rank == i + 3:
        print(f"[Rank {rank}] executando...")

        nx = nx_local
        ny = Gny
        nz = Gnz
        Lx = GLx
        Ly = GLy 
        Lz = GLz

        Y = 10 + 0 * np.random.rand(nx, ny, nz)
        internal_simulpar.BC = DirichletBC3D

        coord, K, psim, dsim, vxsim, vysim, vzsim, divsim = slab3D(internal_simulpar, Y)

        cfl = 1.0
        day = 86400
        tf = 60 * day
        IC = InitialConditionOilStain(20, 80, 20, 80, 0, nz, Lx/nx, Ly/ny, Lz/nz)

        x, y, z, c_hist, dt, nt = run_advection_solver_3D(
            Lx, Ly, Lz, nx, ny, nz,
            ux=vxsim, uy=vysim, uz=vzsim,
            cfl=cfl, tf=tf, IC=IC
        )
        i += 1

####################################################################
# Import libraries
####################################################################
import numpy as np
import sys 
from mpi4py import MPI

#from tools import input_simulation_parameters
from tools import input_simulation_parameters
from tools import DirichletBC3D
from simuladores.slab import slab3D
from tools import run_advection_solver_3D
from tools import InitialConditionOilStain

from tools import troca_de_mensagens_MPI, M_Vizinhos1D, Identifica_Faces, get_local_coordinates, MPI_Vizinhos3D
####################################################################

####################################################################
# Inicializacao MPI
####################################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  

#troca_de_mensagens_MPI(comm, rank, size)
####################################################################

internal_simulpar = input_simulation_parameters('simulation_input2.in')

internal_simulpar.BC = DirichletBC3D
internal_simulpar.Px = 4
internal_simulpar.Py = 1
internal_simulpar.Pz = 2

N_div = internal_simulpar.Px * internal_simulpar.Py * internal_simulpar.Pz

# Variáveis globais
Gnx = internal_simulpar.mesh[0] 
Gny = internal_simulpar.mesh[1] 
Gnz = internal_simulpar.mesh[2]

GLx = internal_simulpar.Dom[0]
GLy = internal_simulpar.Dom[1]
GLz = internal_simulpar.Dom[2]

if Gnx % internal_simulpar.Px == 0:
            nx_local = int (Gnx // internal_simulpar.Px) + 2

if Gny % internal_simulpar.Py == 0:
            ny_local = int (Gny // internal_simulpar.Py) + 2

if Gnz % internal_simulpar.Pz == 0:
            nz_local = int (Gnz // internal_simulpar.Pz) + 2

# x plus := direita, x_minus := esquerda, y_plus := trás, y_minus := frente, z_plus := cima, z_minus := baixo
x_plus = -1
x_minus = -1
y_plus = -1 
y_minus = -1 
z_plus = -1 
z_minus = -1

x_plus, x_minus, y_plus, y_minus, z_plus, z_minus = MPI_Vizinhos3D(Gnx, Gny, Gnz, internal_simulpar.Px, internal_simulpar.Py, internal_simulpar.Pz, comm, rank, size)

for i in range(size):
    if rank == i:
        print(f"{rank}: x: {x_plus, x_minus}, y: {y_plus, y_minus}, z: {z_plus, z_minus}")
####################################################################
# Loop que percorre os processos
####################################################################
for i in range(N_div):
        print(f"[Rank {rank}] executando...")

        nx = nx_local
        ny = ny_local
        nz = nz_local

        Y = 10 + 0 * np.random.rand(nx, ny, nz)

        coord, K, psim, dsim, vxsim, vysim, vzsim, divsim = slab3D(internal_simulpar, Y, rank=rank, size=size)

        cfl = 1.0
        day = 86400
        tf = 60 * day
        IC = InitialConditionOilStain(20, 80, 20, 80, 0, nz, GLx/nx, GLy/ny, GLz/nz)

        x, y, z, c_hist, dt, nt = run_advection_solver_3D(
            GLx, GLy, GLz, nx, ny, nz,
            ux=vxsim, uy=vysim, uz=vzsim,
            cfl=cfl, tf=tf, IC=IC
        )

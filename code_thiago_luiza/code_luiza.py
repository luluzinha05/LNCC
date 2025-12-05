####################################################################
# Import libraries
####################################################################
import numpy as np
from mpi4py import MPI

#from tools import input_simulation_parameters
from tools import input_simulation_parameters
from tools import DirichletBC3D
from simuladores.slab import slab3D
from tools import run_advection_solver_3D
from tools import InitialConditionOilStain

from tools import troca_de_mensagens_MPI, M_Vizinhos1D, Identifica_Faces, get_local_coordinates
####################################################################

####################################################################
# Inicializacao MPI
####################################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  

dest_left, dest_right = M_Vizinhos1D(comm, rank, size)
print(f"Rank {rank}, dest_left {dest_left}, dest_right {dest_right}")

#troca_de_mensagens_MPI(comm, rank, size)
####################################################################

internal_simulpar = input_simulation_parameters('simulation_input2.in')

internal_simulpar.BC = DirichletBC3D
internal_simulpar.Px = 6
internal_simulpar.Py = 1
internal_simulpar.Pz = 1

N_div = internal_simulpar.Px * internal_simulpar.Py * internal_simulpar.Pz

# Variáveis globais
Gnx = internal_simulpar.mesh[0] 
Gny = internal_simulpar.mesh[1] 
Gnz = internal_simulpar.mesh[2]

GLx = internal_simulpar.Dom[0]
GLy = internal_simulpar.Dom[1]
GLz = internal_simulpar.Dom[2]

if Gnx % internal_simulpar.Px == 0:
            nx_local = (Gnx // internal_simulpar.Px) + (1 if (rank == 0 or rank == N_div) else 2)

if Gny % internal_simulpar.Py == 0:
            ny_local = (Gny // internal_simulpar.Py) + (1 if (rank == 0 or rank == N_div) else 2)

if Gnz % internal_simulpar.Pz == 0:
            nz_local = (Gnz // internal_simulpar.Pz) + (1 if (rank == 0 or rank == N_div) else 2) 

face_direita, face_esquerda = Identifica_Faces(rank, size, Gny, Gnz, nx_local)

print(f"[Rank {rank}] Face esquerda -> {face_esquerda}")
print(f"[Rank {rank}] Face direita  -> {face_direita}")

####################################################################
# Loop que percorre os 4 processos
####################################################################
i = 0

for _ in range(N_div):

    if rank == 0:
        print(f"[Rank {rank}] executando...")

        nx = nx_local
        ny = ny_local
        nz = nz_local
     
        Y = 10 + 0 * np.random.rand(nx, ny, nz)

        coord, K, psim, dsim, vxsim, vysim, vzsim, divsim = slab3D(internal_simulpar, Y, rank=rank, size=size)


        for j in range(ny * nz):
            print(f"Rank {rank} ID da face da direita {face_direita[j]} Coordenadas {get_local_coordinates(face_direita[j], coord)}")

        cfl = 1.0
        day = 86400
        tf = 60 * day
        IC = InitialConditionOilStain(20, 80, 20, 80, 0, nz, GLx/nx, GLy/ny, GLz/nz)

        x, y, z, c_hist, dt, nt = run_advection_solver_3D(
            GLx, GLy, GLz, nx, ny, nz,
            ux=vxsim, uy=vysim, uz=vzsim,
            cfl=cfl, tf=tf, IC=IC
        )
        i += 1

    elif rank == N_div:
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
        i += 1

    else:
        print(f"[Rank {rank}] executando...")

        nx = nx_local
        ny = ny_local
        nz = nz_local

        Y = 10 + 0 * np.random.rand(nx, ny, nz)

        coord, K, psim, dsim, vxsim, vysim, vzsim, divsim = slab3D(internal_simulpar, Y)

        cfl = 1.0
        day = 86400
        tf = 60 * day
        IC = InitialConditionOilStain(20, 80, 20, 80, 0, nz, GLx/nx, GLy/ny, GLz/nz)

        x, y, z, c_hist, dt, nt = run_advection_solver_3D(
            GLx, GLy, GLz, nx, ny, nz,
            ux=vxsim, uy=vysim, uz=vzsim,
            cfl=cfl, tf=tf, IC=IC
        )
        i += 1

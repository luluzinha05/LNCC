#Exemplo 1D
#####################################################################################
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

dims = [size]
periods = [False]
cart = comm.Create_cart(dims, periods=periods, reorder=False)

coord = cart.Get_coords(rank)[0]

# Descobre vizinhos
src_left, dest_right = cart.Shift(0, +1)
src_right, dest_left = cart.Shift(0, -1)

# Usando arrays NumPy (necessário para Sendrecv)
send_value = np.array(rank, dtype='i')   # tipo inteiro
recv_value = np.array(-1, dtype='i')

cart.Sendrecv(
    sendbuf=send_value, dest=dest_right, sendtag=0,
    recvbuf=recv_value, source=src_left, recvtag=0
)

print(f"Rank {rank} (coord={coord}) recebeu {recv_value[()]} da esquerda ({src_left})")

#####################################################################################
#Exemplo 2D
#####################################################################################

# from mpi4py import MPI
# import numpy as np

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

# dims = MPI.Compute_dims(size, 2)
# periods = [False, False]
# cart = comm.Create_cart(dims, periods=periods, reorder=True)

# coords = cart.Get_coords(rank)
# i, j = coords

# ny, nx = 4, 4
# field = np.full((ny+2, nx+2), rank, dtype=np.float64)

# src_up, dst_up   = cart.Shift(0, -1)
# src_down, dst_down = cart.Shift(0, +1)
# src_left, dst_left = cart.Shift(1, -1)
# src_right, dst_right = cart.Shift(1, +1)

# # Troca horizontal
# send_right = field[1:-1, nx].copy()
# recv_left = np.empty_like(send_right)
# if dst_right != MPI.PROC_NULL and src_left != MPI.PROC_NULL:
#     cart.Sendrecv(sendbuf=send_right, dest=dst_right, sendtag=0,
#                   recvbuf=recv_left, source=src_left, recvtag=0)
#     field[1:-1, 0] = recv_left

# send_left = field[1:-1, 1].copy()
# recv_right = np.empty_like(send_left)
# if dst_left != MPI.PROC_NULL and src_right != MPI.PROC_NULL:
#     cart.Sendrecv(sendbuf=send_left, dest=dst_left, sendtag=1,
#                   recvbuf=recv_right, source=src_right, recvtag=1)
#     field[1:-1, nx+1] = recv_right

# # Troca vertical
# send_down = field[ny, 1:-1].copy()
# recv_up = np.empty_like(send_down)
# if dst_down != MPI.PROC_NULL and src_up != MPI.PROC_NULL:
#     cart.Sendrecv(sendbuf=send_down, dest=dst_down, sendtag=2,
#                   recvbuf=recv_up, source=src_up, recvtag=2)
#     field[0, 1:-1] = recv_up

# send_up = field[1, 1:-1].copy()
# recv_down = np.empty_like(send_up)
# if dst_up != MPI.PROC_NULL and src_down != MPI.PROC_NULL:
#     cart.Sendrecv(sendbuf=send_up, dest=dst_up, sendtag=3,
#                   recvbuf=recv_down, source=src_down, recvtag=3)
#     field[ny+1, 1:-1] = recv_down

# print(f"[Rank {rank}] coords={coords}\n{field}\n", flush=True)

###############################################################################
import sys
from numpy.linalg import solve
from scipy.sparse.linalg import spsolve
from tools import coordinates1D, coordinates2D, coordinates3D_MPI, compute_perm
from tools import compute_trans_1D, setup_system_1D
from tools import compute_trans_2D, setup_system_2D
from tools import compute_trans_3D, setup_system_3D
from tools import compute_tpfa_velocity1D, compute_div1D
from tools import compute_tpfa_velocity2D, compute_div2D
from tools import compute_tpfa_velocity3D, compute_div3D
import numpy as np
#import time 
#from tools import format_seconds_to_hhmmss
#from tools import plot_pres1D, plot_vel_div1D, plot_pres2D, plot_vel_div2D
from numpy import max #, sum, abs
###############################################################################

###############################################################################
def slab1D(inputpar,fieldY=0.0,tolerance=1e-6):
    '''Solve the Darcy equation via finite differences'''
    ##########################################################
    # Define the grid ========================================
    nx = inputpar.mesh[0]
    Lx = inputpar.Dom[0]
    dx = Lx / nx
    _, coord = coordinates1D(nx, Lx)
    ##########################################################
    beta= inputpar.beta
    rho = inputpar.rho
    ##########################################################
    # Permeability field (heterogeneous) =====================
    K = compute_perm(fieldY, rho, beta)

    # Calcular transmissibilidades (apenas N+1 valores)
    S = 1
    mu  = inputpar.mu
    trans = compute_trans_1D(K, dx, S, mu)
    
    # Montar o sistema de equações
    PL = inputpar.PL
    PR = inputpar.PR
    BC = inputpar.BC(PL,PR)
    q = inputpar.q if hasattr(inputpar, 'q') else None
    A, b = setup_system_1D(trans, BC, q)

    # Solve the linear system
    p_flat = solve(A, b)
    p = p_flat.reshape((nx,), order='F')
    
    # save the pressure field ================================
    pos = inputpar.positions
    data = p[pos]
    
    v = compute_tpfa_velocity1D(p, trans, BC)
    div = compute_div1D(v)
    v = v/S

    if max(div) > tolerance:
        print('Warning: Divergence is not zero! Max div =', max(div))

    return coord, K, p_flat, data, v, div
###############################################################################

###############################################################################
def slab2D(inputpar,fieldY=0.0,tolerance=1e-6):
    '''Solve the Darcy equation via finite differences'''
    ##########################################################
    # Define the grid ========================================
    nx,ny = inputpar.mesh[0], inputpar.mesh[1]
    Lx, Ly = inputpar.Dom[0], inputpar.Dom[1]
    dx = Lx / nx
    dy = Ly / ny

    _, coord = coordinates2D(nx, ny, Lx, Ly)
    ##########################################################
    beta= inputpar.beta
    rho = inputpar.rho
    ##########################################################
    # Permeability field (heterogeneous) =====================
    K = compute_perm(fieldY, rho, beta)

    # Calcular transmissibilidades (apenas N+1 valores)
    mu  = inputpar.mu
    H = 1

    Tx, Ty = compute_trans_2D(K, dx, dy, H, mu)

    # Montar o sistema de equações
    PL = inputpar.PL
    PR = inputpar.PR
    BC = inputpar.BC(PL,PR)
    
    q = inputpar.q if hasattr(inputpar, 'q') else None
    A, b = setup_system_2D(Tx, Ty, BC, q)

    # Solve the linear system
    p_flat = spsolve(A, b)
    p = p_flat.reshape((nx,ny), order='F')
    
    # save the pressure field ================================
    pos = inputpar.positions
    data = p_flat[pos]
    
    vx,vy = compute_tpfa_velocity2D(p, Tx, Ty, BC)
    div = compute_div2D(vx,vy)
    #div = div/abs(sum(div) - (sum(q) if q is not None else 0.0))
    vx = vx/(H*dy) # fluxo volumetrico em x -> velocidade em x
    vy = vy/(H*dx) # fluxo volumetrico em y -> velocidade em y

    if max(div) > tolerance:
        print('Warning: Divergence is not zero! Max div =', max(div))

    return coord, K, p, data, vx, vy, div
###############################################################################

###############################################################################
def slab3D(inputpar,fieldY=0.0,tolerance=1e-6, rank=0, size=1):
    '''Solve the Darcy equation via finite differences'''
    ##########################################################
    # Define the grid ========================================
    nx,ny,nz = inputpar.mesh[0], inputpar.mesh[1], inputpar.mesh[2]
    Lx, Ly, Lz = inputpar.Dom[0], inputpar.Dom[1], inputpar.Dom[2]
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    idx, coord = coordinates3D_MPI(nx, ny, nz, Lx, Ly, Lz,
                                   inputpar.mesh[0], inputpar.mesh[1], inputpar.mesh[2],   # Gnx, Gny, Gnz
                                   inputpar.Dom[0], inputpar.Dom[1], inputpar.Dom[2],      # GLx, GLy, GLz
                                   rank, size
                                  )
    
    #imprimir o cord aposimplementaçãodas ghostcells (for i < numerode processos    if rank==i  print processo i    imprimir a variavel cord)
    for i in range(size):
        if rank == i:
            print(f"\n[Rank {rank}] Coordenadas locais (com ghost cells):")
            print(coord)
    ##########################################################
    beta= inputpar.beta
    rho = inputpar.rho
    ##########################################################
    # Permeability field (heterogeneous) =====================
    K = compute_perm(fieldY, rho, beta)

    # Calcular transmissibilidades (apenas N+1 valores)
    mu  = inputpar.mu
    
    # Transmissibilidade em x, y e z
    Tx, Ty, Tz = compute_trans_3D(K, dx, dy, dz, mu)

    # Montar o sistema de equações
    PL = inputpar.PL
    PR = inputpar.PR
    BC = inputpar.BC(PL,PR)
    
    # Se houver fonte, monta o vetor de fontes
    q = inputpar.q if hasattr(inputpar, 'q') else None
    A, b = setup_system_3D(Tx, Ty, Tz, BC, q)

    # Solve the linear system
    p_flat = spsolve(A, b)
    p = p_flat.reshape((nx,ny,nz), order='F')
    
    # save the pressure field ================================
    pos = inputpar.positions
    data = p_flat[pos]
    
    vx,vy,vz = compute_tpfa_velocity3D(p, Tx, Ty, Tz, BC)
    div = compute_div3D(vx,vy,vz)
    #div = div/abs(sum(div) - (sum(q) if q is not None else 0.0))
    vx = vx/(dy*dz) # fluxo volumetrico em x -> velocidade em x
    vy = vy/(dz*dx) # fluxo volumetrico em y -> velocidade em y
    vz = vz/(dx*dy) # fluxo volumetrico em y -> velocidade em y

    if max(div) > tolerance:
        print('Warning: Divergence is not zero! Max div =', max(div))

    return coord, K, p, data, vx, vy, vz, div
###############################################################################

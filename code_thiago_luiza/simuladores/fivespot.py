###############################################################################
import sys
from numpy.linalg import solve, norm
from scipy.sparse.linalg import spsolve
from tools import coordinates1D, coordinates2D, compute_perm
from tools import compute_trans_1D, setup_system_1D
from tools import compute_trans_2D, setup_system_2D
from tools import compute_tpfa_velocity1D, compute_div1D
from tools import compute_tpfa_velocity2D, compute_div2D
import time 
from tools import format_seconds_to_hhmmss
from tools import plot_pres1D, plot_vel_div1D, plot_pres2D, plot_vel_div2D
from numpy import sum, abs, max
###############################################################################

###############################################################################
def fivespot2D(inputpar,fieldY=0.0,flag=True,tolerance=1e-6,fact=1):
    '''Solve the Darcy equation via finite differences'''
    ##########################################################
    # Define the grid ========================================
    nx,ny = inputpar.mesh[0], inputpar.mesh[1]
    Lx, Ly = inputpar.Dom[0], inputpar.Dom[1]
    dx = Lx / nx
    dy = Ly / ny

    idx, coord = coordinates2D(nx, ny, Lx, Ly)
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

    if flag:
        plot_pres2D(coord,pos,nx,ny,K,p,data)

        i = 1/max([norm(vx),norm(vy)])
        plot_vel_div2D(i*vx,i*vy,coord,div,nx,ny,Lx,Ly,dx,dy,fact,1)

    return p, data, vx, vy
###############################################################################

###############################################################################
def fivespot3D(inputpar,fieldY=0.0,flag=True,tolerance=1e-6,fact=1):
    '''Solve the Darcy equation via finite differences'''
    ##########################################################
    # Define the grid ========================================
    nx,ny,nz = inputpar.mesh[0], inputpar.mesh[1], inputpar.mesh[2]
    Lx, Ly, Lz = inputpar.Dom[0], inputpar.Dom[1], inputpar.Dom[2]
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    idx, coord = coordinates3D(nx, ny, nz, Lx, Ly, Lz)
    ##########################################################
    beta= inputpar.beta
    rho = inputpar.rho
    ##########################################################
    # Permeability field (heterogeneous) =====================
    K = compute_perm(fieldY, rho, beta)

    # Calcular transmissibilidades (apenas N+1 valores)
    mu  = inputpar.mu
    
    Tx, Ty, Tz = compute_trans_3D(K, dx, dy, dz, mu)

    # Montar o sistema de equações
    PL = inputpar.PL
    PR = inputpar.PR
    BC = inputpar.BC(PL,PR)
    
    q = inputpar.q if hasattr(inputpar, 'q') else None
    A, b = setup_system_3D(Tx, Ty, Tz, BC, q)

    # Solve the linear system
    p_flat = spsolve(A, b)
    p = p_flat.reshape((nx,ny), order='F')
    
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

    if flag:
        plot_pres3D(coord,pos,nx,ny,nz,K,p,data)

        i = 1/max([norm(vx),norm(vy),norm(vz)])
        plot_vel_div3D(i*vx,i*vy,i*vz,coord,div,nx,ny,nz,Lx,Ly,Lz,dx,dy,dz,fact,1)

    return p, data, vx, vy, vz
###############################################################################

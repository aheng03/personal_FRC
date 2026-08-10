# Author: Adelina Hengyucius
# Date:  6.24.26
# Shape Index Functions 
### --- IMPORTS
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import sympy as sp
from sympy.abc import x

from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]  # goes up from FRC to code
sys.path.append(str(base_dir))
sys.path.append("Grad_Shaf")


from _functions4plasma import *
from _plottingParameters import * 
from _linerParameters import *
from Grad_Shaf._gs_im import *


def poly_fit_eq(rSep, zSep, d):
    '''
    Polynomial fit for rSep, ySep data; alternative to standardized separatrix shape fit 
    '''
    coeff = np.polyfit(rSep, zSep, d)
    equation = sp.Poly(coeff, x).as_expr()
    func = sp.lambdify(x, equation, modules='numpy')    # standard Python function 
    return equation, func


def z_cap(r, N, a_v, E_v, eps, Xs):
    '''
    Section of ellipsoid near the end regions of the separatrix
    '''
    R_max = Xs*Rw       # equal to Rs -- separatrix radius 
    R_min = (-R_max*(eps-1))/(eps+1)
    R0 = (R_max + R_min) / 2 
    
    a = a_v/R0
    b = E_v*a
    r = np.array(r)
    return b - N*a + N*a*np.sqrt(1-(r/a)**2)


def least_squares_fit_eps(N, eq_func, a, E, eps, Xs):
    '''
    minimize the following quantity to solve for N
    eps(N) = integral of the following integrand
    
    y1 = separatrix shape polynomial fit
    y2 = ellipsoidal fit at end regions of the sepparatrix 
    integrand = r*([y1 - y2]^2)
    '''
    R_max = Xs*Rw       # equal to Rs -- separatrix radius 
    R_min = (-R_max*(eps-1))/(eps+1)
    R0 = (R_max + R_min) / 2 
    
    a = a/R0
    b = E*a

    start = 0
    end = 0.8*a

    def integrand(x_val):
        y1 = eq_func(x_val)
        y2 = b - N*a + N*a*np.sqrt(1-(x_val/a)**2)
        return x_val * (y1 - y2)**2
    
    result, error = spi.quad(integrand, start, end)
    return result


def gs_(elong, eps, delta, Xs):
    R_max = Xs*Rw       # equal to Rs -- separatrix radius 
    R_min = (-R_max*(eps-1))/(eps+1)
    R0 = (R_max + R_min) / 2 
    a = (R_max - R_min) / 2

    kap = 2*elong

    ### === DOMAIN SETUP
    domx = 1.1                                 # weight to extend the domain; helps show psi=1 curves better
    Rd = Rw
    Zd = kap*a * domx
    dr = 0.005/Lconv                           # mesh fidelity in r-dir [m]
    dz = 0.005/Lconv                           # mesh fidelity in z-dir [m]
    r = np.arange(-Rd, Rd+dr, dr)               #r-array for mesh
    z = np.arange(-Zd, Zd+dz, dz)               #z-array for mesh   
    r = np.delete(r, np.where(r==0))
    z = np.delete(z, np.where(z==0))
    x_ar = r/R0
    y_ar = z/R0
    R, Z = np.meshgrid(x_ar, y_ar, indexing='ij')     #(R,Z) mesh elements

    flux_val = calculate_flux(True, True, False, 0, 1, kap, delta, R, Z)
    
    # --- VISUAL PREVIEW ---
    # numColors = 11
    # flux_min = flux_val.min()
    # levels = np.linspace(flux_min, 0, numColors)
    # fig, ax = plt.subplots(dpi=100)
    # ax.contour(R, Z, flux_val, levels=levels, cmap='rainbow')
    # ax.set_aspect('equal')
    # plt.show()

    # --- EXTRACTION ---
    # 1) Build an OFF‐SCREEN figure, targeting the 0-flux boundary
    plt.ioff()                     
    fig, ax = plt.subplots(dpi=50)  
    
    # Target value we are hunting for
    target_level = 0.0 
    cs = ax.contour(R, Z, flux_val, levels=[target_level])
    plt.close(fig)                 

    # 2) Now extract the contour segments safely using the target_level value
    try:
        idx = list(cs.levels).index(target_level) # Fixed: looking for 0.0, not the full matrix
    except ValueError:
        raise RuntimeError(f"No ψ={target_level} level found in cs.levels={cs.levels}")
        
    segs = cs.allsegs[idx]
    if not segs:
        # Fallback: If 0.0 exact is missing due to floats, try slightly below zero
        plt.ioff()
        fig, ax = plt.subplots(dpi=50)
        cs = ax.contour(R, Z, flux_val, levels=[-1e-5])
        plt.close(fig)
        segs = cs.allsegs[0]
        
        if not segs:
            raise RuntimeError("No plasma boundary/separatrix (flux=0) contour found in the domain.")

    # 3) Convert each Nx2 array into (r_i, z_i) loops
    loops = []
    for seg in segs:
        verts = np.asarray(seg)    # shape=(Npts,2)
        r_i = verts[:, 0].copy()
        z_i = verts[:, 1].copy()
        loops.append((r_i, z_i))

    # Safely return the longest closed loop (the main plasma boundary)
    longest = max(loops, key=lambda pair: pair[0].shape[0])
    
    return longest


def find_rz(rSep, zSep, a, Xs, eps):
    '''
    Getting rid of z-values less than 0
    '''
    R_max = Xs*Rw       # equal to Rs -- separatrix radius 
    R_min = (-R_max*(eps-1))/(eps+1)
    R0 = (R_max + R_min) / 2 

    # Since the flux is normalized against the R0, a must also be normalized
    a = a/R0
    mask = (zSep > 0) & (np.abs(rSep) < a)
    r_arr = rSep[mask]
    z_arr = zSep[mask]
    return r_arr, z_arr


def get_flux_contours(psi, R, Z, psi_level_mag, return_all=False):
    # 1) Build a tiny OFF‐SCREEN figure, extract contours, then close it:
    plt.ioff()                     # turn off interactive showing
    fig, ax = plt.subplots(figsize=(0.1,0.1))  
    cs = ax.contour(R, Z, psi, levels=[psi_level_mag])
    plt.close(fig)                 # immediately close it so nothing pops up

    # 2) Now extract the contour segments from cs:
    try:
        idx = list(cs.levels).index(psi_level_mag)
    except ValueError:
        raise RuntimeError(f"No ψ={psi_level_mag} level found in cs.levels={cs.levels}")
    segs = cs.allsegs[idx]
    if not segs:
        raise RuntimeError(f"No ψ={psi_level_mag} contour found in the domain.")

    # 3) Convert each Nx2 array into (r_i, z_i) loops exactly as you had:
    loops = []
    for seg in segs:
        verts = np.asarray(seg)    # shape=(Npts,2)
        r_i = verts[:,0].copy()
        z_i = verts[:,1].copy()
        loops.append((r_i, z_i))

    if return_all:
        return loops

    longest = max(loops, key=lambda pair: pair[0].shape[0])
    return longest


### --- IMPORTS
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
from sympy.abc import x
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.optimize import root 

from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]  # goes up from FRC to code
sys.path.append(str(base_dir))

from _functions4plasma import *

### === Shape Index
def shape_index(a, b, K):
    return (K*b**2)/(a)


# Eq 4a-4d from Steinhaurer 1992
def psi_stein_N(r, z, a, b, N): 
    '''
    Tilt stability of a gyroviscous field-reversed configuration
    with realistic equilibria paper definition
    
    r: 2D np.array
    z: 2D np.array
    a: float; separatrix radius
    b: float; separatrix half-length
    N: float; shape index
    '''
    r_ = r/a
    z_ = z/a

    E = b/a
    D0 = (8*E**4 - 1) / (8*E**4 + 4*E**2)
    D1 = (4*E**2 + 1) / (8*E**4 + 4*E**2)
    B1 = (1-N) / (4*E**2*(1+2*D1) + N*(1+D1))

    g = 1 - r_**2 - (z_ / E)**2 + B1*(1-D0*(r_**2 - 4*z_**2) - D1*(r_**4 - 12*r_**2*z**2 + 8*z_**4))
    
    return 0.5*r**2*g


### === PARAMETER FUNCTIONS
def pressure_jeff_mesh(R, Z, Jphi, Br, Bz):
    '''
    frc05_2d
    '''
    """
    Compute pressure P(r,z) on a 2D (R,Z) mesh using Jeff’s method:

      ∂P/∂r = Jphi * Bz
      ∂P/∂z = -Jphi * Br    (not used for integration here)

    We enforce P(r_max, z) = 0 at the outermost r, and integrate inward
    along each fixed‐z column.  

    Inputs:
      - R    : 2D array, shape (Nr, Nz), radial mesh (meters).  Must satisfy R[i,j] = r_i.
      - Z    : 2D array, shape (Nr, Nz), axial  mesh (meters).  Must satisfy Z[i,j] = z_j.
      - Jphi : 2D array, shape (Nr, Nz), J_φ at each (r_i, z_j) [A/m²].
      - Br   : 2D array, shape (Nr, Nz), B_r(r_i, z_j) [T].
      - Bz   : 2D array, shape (Nr, Nz), B_z(r_i, z_j) [T].

    Returns:
      - P : 2D array, shape (Nr, Nz), pressure P(r_i, z_j) [Pa], with P(r_max, z_j)=0.
    """

    Nr, Nz = R.shape

    # 1) Extract the 1D r‐axis from the first column of R:
    #    We assume R was created via meshgrid(r, z, indexing='ij'),
    #    so R[i,j] == r[i].  Thus r = R[:,0].
    r_1d = R[:, 0].copy()       # shape = (Nr,)

    # 2) Compute ∂P/∂r = Jphi * Bz for every mesh point:
    dP__dr = Jphi * Bz          # shape = (Nr, Nz)

    # (We also could form dP__dz = -Jphi * Br, but it's not needed for the radial integration.)

    # 3) Allocate the output pressure array, initially zeros:
    P = np.zeros_like(Jphi)     # shape = (Nr, Nz)

    # 4) Build a reversed‐r array so that index‐0 corresponds to r_max:
    r_rev = r_1d[::-1]          # shape = (Nr,) from largest r to smallest

    # 5) Loop over each z‐column (fixed j).  Integrate dP/dr from r_max → r=0:
    for j in range(Nz):
        # 5a) Extract column j of dP/dr, then reverse it in r:
        col_dPdr     = dP__dr[:, j]     # shape = (Nr,)
        col_dPdr_rev = col_dPdr[::-1]    # now r_rev[0] is the maximum r

        # 5b) Cumulative trapezoidal integration along reversed‐r axis.
        #     cumtrapz(y, x, initial=0) returns an array Y of length len(x),
        #     where Y[k] = ∫_{x[0]}^{x[k]} y(x') dx'.  Here x[0] = r_rev[0] = r_max.
        integral_rev = cumtrapz(col_dPdr_rev, r_rev, initial=0.0)
        # By construction, integral_rev[0] = 0 at r_rev[0]=r_max, so P(r_max)=0.

        # 5c) Flip back to original r‐ordering and negate to get P(r_i):
        #     integral_rev[::-1][i] = ∫_{r_i}^{r_max} dP/dr' dr'.
        #     We want P(r_i) = - ∫_{r_i}^{r_max} (dP/dr') dr'  (so that ∂P/∂r = dP/dr).
        P[:, j] = - integral_rev[::-1]
    return P


def pressure_sporer(Bw, Xs, psi_int, a, b, f):
    eps = a / b
    Pe = Bw**2 / (2*mu0)
    x = 3/2 * Xs**2
    y = np.sqrt(3/2) * Xs * 8 * psi_int / (Bw*a**2)
    z = f * eps**2

    P = Pe * (1 - x + y + z)
    return P


def sporer_Be(Xs, B0):
    Be = B0 / (1 - Xs**2)
    return Be


### === EXTERNAL FUNCTIONS 
def external_dpsi__dr(r, z, Bw, a, b, E0, E1, E2, alp):
    '''
    '''
    denom1 = (alp*b-z)**2 + r**2
    denom2 = (alp*b+z)**2 + r**2
    T0 = E0 * r
    T1 = E1 * (r/a**2) * (2*r**2 - 4*z**2)
    T2 = E2 * (a**2/2) * (r*(alp*b-z)/(denom1**1.5) + r*(alp*b+z)/(denom2**1.5))

    dpsi__dr = Bw * (T0 + T1 - T2)
    return dpsi__dr


def external_dpsi__dz(r, z, Bw, a, b, E0, E1, E2, alp):
    denom1 = (alp*b-z)**2 + r**2
    denom2 = (alp*b+z)**2 + r**2
    T1 = E1 * 8*z*r**2/a**4
    T2 = E2 * ((z-alp*b)**2/(denom1**1.5) - 1/denom1**0.5 + 1/denom2**0.5 - (alp*b+z)**2/denom2**1.5)
    
    dpsi__dz = Bw * a**2/2 * (T2 - T1)
    return dpsi__dz


def external_Be(Bw, a, b, Xs):
    Be__Bw = 1 + 1.46 * (3.2 + ((b/a) / (1.5-Xs))**4)**(-1)
    Be = Bw * Be__Bw
    return Be


def external_psi(r, z, Bw, a, b, E0, E1, E2, alp):
    psi_w = Bw * a**2 / 2
    e0 = (r/a)**2
    e1 = (r/a)**2 * ((r/a)**2 - 4*(z/a)**2)
    denom1 = (alp*b+z)**2 + r**2
    denom2 = (alp*b-z)**2 + r**2
    e2 = (alp*b+z)/np.sqrt(denom1) + (alp*b-z)/np.sqrt(denom2)
    
    psi = psi_w * (E0*e0 + E1*e1 + E2*e2)
    return psi


def external_E_params(Es, eps, Xs, sig):
    E0, E1, E2, a = Es

    eq1 = E0 - E1 * (4/eps**2) + E2 * (2*eps**2*a/(1-a**2)**2)
    eq2 = E0 + E1 + 2*a*E2 / np.sqrt(eps**2+a**2)
    e0 = eps**2*Xs**3*E2/4/sig**2 
    e1 = (a+sig) / np.sqrt(eps**2 + Xs**2*(a+sig)**2)
    e2 = (a-sig) / np.sqrt(eps**2 + Xs**2*(a-sig)**2)
    e3 = 2*a / np.sqrt(eps**2 + (Xs*a)**2)
    eq3 = E1 - e0 * (e1 + e2 - e3)
    eq4 = E0 + 2*E1/Xs**2 - eps**2*a*Xs**3*E2 / ((a*Xs)**2 + eps**2)**(1.5) - 1

    return [eq1, eq2, eq3, eq4]


### === INTERNAL FUNCTIONS 
def internal_psi_sporer(r, z, a, b, Be, Xs, f):     #internal flux
    '''
    r:  np.array (2D)
    z:  np.array (2D)
    a:  float; separatrix radius 
    b:  float; separatrix half length
    Be: int; external magnetic field 
    Xs: float; normalized separatrix ratio 
    f:  arbitrary constant = 1.5 
    '''
    eps = a / b         #Inverse elongation
    T1 = np.sqrt(3/2)
    T2 = (Xs * Be * r**2) / 2
    T3 = 1 - (r/a)**2 - (z/b)**4 + f * eps**2
    
    psi_int = T1 * T2 * T3 
    return psi_int


def internal_psi_stein(r, z, a, b, Bw, Rc):
    Xs = a/Rc
    eps = a/b
    Eguess = [Bw, Bw/3, Bw/5, 0.9]

    sol = root(                                                     # solves the system of external E equations
    fun=external_E_params,                                          # designates the function
    x0=Eguess,                                                      # initial guesses for E0, E1, E2, alpha
    args=(eps, Xs, sig),                               # designates these parameters as constants
    method='lm',                                                    # same algorithm as old fsolve, or try 'lm'
    tol=1e-6                                                        # tolerance
    )
    
    E0, E1, E2, alpha = sol.x
    
    T1 = Bw*a**2/2
    T2 = E0*r**2/a**2 + E1*(r**4/a**4-4*z**2*r**2/a**4)
    T3 = E2*((alpha*b+z)/np.sqrt(r**2+(alpha*b+z)**2)+(alpha*b-z)/np.sqrt(r**2+(alpha*b-z)**2))
    
    return T1*(T2+T3)


def internal_dpsi__dr_stein(r, z, a, b, B0, B1, D0, D1):
    b0 = 1 - 2*(r/a)**2 - (z/b)**2
    d0 = 2 * (r/a)**2 - 4 * (z/a)**2
    d1 = 3 * (r/a)**4 - 24 * (r*z/a/a)**2 + 8 * (z/a)**4
    b1 = 1 + D0*d0 + D1*d1

    dpsi__dr = (B0 * r * b0) + (B1 * r * b1)
    return dpsi__dr


def internal_dpsi__dz_stein(r, z, a, b, B0, B1, D0, D1):
    b0 = - 2 * z / b**2
    d0 = - 8 * z / a**2
    d1 = -24 * r**2 * z / a**4 + 32 * z**3 / a**4
    b1 = D0 * d0 + D1 * d1
    
    dpsi__dz = (r**2 / 2) * (B0*b0 + B1*b1)
    return dpsi__dz


def internal_dpsi__dr_sporer(r, z, a, b, Be, Xs, f):
    eps = a / b
    F = np.sqrt(3/2) * Xs * Be * r
    P = 1 - 2*(r/a)**2 - (z/b)**4 + f * eps**2 
    dpsi__dr = F * P
    return dpsi__dr
    

def internal_dpsi__dz_sporer(r, z, a, b, Be, Xs):
    dpsi__dz = - np.sqrt(6) * Xs * Be * r**2 * z**3 / b**4
    return dpsi__dz


def internal_D0(eps):
    D0 = (eps**4-8) / (4*(2+eps**2))
    return D0


def internal_D1(eps):
    D1 = - (eps**2*(4+eps**2)) / (4*(2+eps**2))
    return D1


def internal_B1(B0, D0, D1, N, eps):
    T2 = eps**2 * (N-1)
    T0 = D0 * (4+eps**2*N)
    T1 = D1 * (12 + 2*eps**2 * N)

    B1 = B0 * T2 / (T0 + T1)
    return B1


def internal_B0(a, b, Xs, Bw):
    N = 2.56 - 3.28*Xs
    D = 9.65 - 15.2*Xs + (b/a)**4
    p = 1 + N/(Xs**2 * D)    
    sr = np.sqrt(3/2)*Xs

    B0 = Bw * sr * p
    return B0
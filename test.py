### === IMPORTS
from func_steinhaurer import *
from _functions4plasma import *

import math
import numpy as np 
import sympy as sp 
from scipy.optimize import root
import matplotlib.pyplot as plt 
import sys

### === Finding curvature function (as per curvature.ipynb)
# Define Variables
E0, E1, E2, E3 = sp.symbols("E:4")
B, r, z, a, b = sp.symbols('B r z a b')
psi = sp.symbols('P')

# Define Equations (for Eq. 2)
T1 = E0*(r**2/a**2)
T2 = E1*(r**2/a**2)*(r**2/a**2 - 4*z**2/a**2)
T3 = E2*(((E3*b + z)/((r**2 + (E3*b+z)**2)**0.5))+((E3*b - z)/((r**2 + (E3*b-z)**2)**0.5)))

psi = sp.simplify((B*a**2)/2*(T1+T2+T3))
psi_r = sp.diff(psi, r)
psi_rr = sp.diff(psi_r, r)
psi_z = sp.diff(psi, z)
psi_zz = sp.diff(psi_z, z)
psi_rz = sp.diff(psi_r, z)

# Define Functions
f_psi = sp.lambdify((B, E0, E1, E2, E3, r, z, a, b), psi)
f_psi_r = sp.lambdify((B, E0, E1, E2, E3, r, z, a, b), psi_r)
f_psi_rr = sp.lambdify((B, E0, E1, E2, E3, r, z, a, b), psi_rr)
f_psi_z = sp.lambdify((B, E0, E1, E2, E3, r, z, a, b), psi_z)
f_psi_zz = sp.lambdify((B, E0, E1, E2, E3, r, z, a, b), psi_zz)
f_psi_rz = sp.lambdify((B, E0, E1, E2, E3, r, z, a, b), psi_rz)


# Curvature based of Steinhaurer's analytical equation (Eq. 2)
def curvature(B, E0, E1, E2, E3, r, z, a, b):
    num = f_psi_rr(B, E0, E1, E2, E3, r, z, a, b)*(f_psi_z(B, E0, E1, E2, E3, r, z, a, b))**2 - 2*f_psi_rz(B, E0, E1, E2, E3, r, z, a, b)*f_psi_r(B, E0, E1, E2, E3, r, z, a, b)*f_psi_z(B, E0, E1, E2, E3, r, z, a, b) + f_psi_zz(B, E0, E1, E2, E3, r, z, a, b)*(f_psi_r(B, E0, E1, E2, E3, r, z, a, b))**2
    den = ((f_psi_r(B, E0, E1, E2, E3, r, z, a, b))**2 + (f_psi_z(B, E0, E1, E2, E3, r, z, a, b))**2)**(1.5)
    return np.abs(num/den)


### === Definitions
Lconv = 1e3
acc = 100
# Parameters / Domain Definitions 
Rw          = 6.1/Lconv                 # Wall Radius [m]
Rc          = Rw
zLen        = 200/Lconv                 # Liner length [m]

B0  = 30
sig = f = 1.5                           # Flare parameter; adjustable parameter

T           = np.array([50])            # Temp in [eV]
TK          = T * eV/kb                 # Temp number in [K]


### === 2D Mesh
Nr, Nz      = 1000, 1000
Rmax        = Rw                        # maximum r-domain value [m]
Rmin        = 0                         # minimum r-domain value [m]
dr          = (Rmax - Rmin) / Nr        # radial differential [m]
Zmax        = zLen / 2                  # maximum z-domain value [m]
Zmin        = -zLen / 2                 # minimum z-domain value [m]
dz          = (Zmax - Zmin) / Nz        # axial differential [m]
domx        = 1.4                       # domain multiplier to make plotting modifications easier
domy        = 0.8                       # domain multiplier to make plotting modifications easier
Rd          = Rmax * domx               # domain radius [m]
Zd          = Zmax * domy               # domain z-length [m]

r           = np.arange(-Rd, Rd+dr, dr)
z           = np.arange(-Zd, Zd+dz, dz)    
r_mesh, z_mesh = np.meshgrid(r, z, indexing='ij')
Nr, Nz      = r_mesh.shape


### === Define Arrays 
conI        = True                      # varying elongation and varying Xs
conII       = True                     # varying elongation and constant Xs
conIII      = False                     # constant elongation and varying Xs 

### === Calculating Shape Index Values
mask                = list(range(0, acc))
N_arr               = []
e_params            = []

Br_ext_arr          = np.zeros((acc, Nr, Nz))
Br_int_arr          = np.zeros((acc, Nr, Nz))
Bz_ext_arr          = np.zeros((acc, Nr, Nz))
Bz_int_arr          = np.zeros((acc, Nr, Nz))

if conI:
    print("Varying Elongation and Varying Xs")
    elong_arr   = np.linspace(1, 10,num = acc)
    Xs_arr      = np.linspace(0.3, 0.9,num = acc)
    a_arr       = Rc*Xs_arr
    b_arr       = np.multiply(a_arr, elong_arr)
    Bw_arr      = B0/(1-(Xs_arr**2))

    # === Finding E_n
    for i, E in enumerate(elong_arr):
        Xs = Xs_arr[i]
        Bw = Bw_arr[i]
        a  = a_arr[i]
        b  = b_arr[i]
        initial_guess = [Bw, Bw/3, Bw/5, 0.9]

        Br_int_arr[i] = (1/r_mesh) * internal_dpsi__dz_sporer(r_mesh, z_mesh, a, b, Bw, Xs)
        Bz_int_arr[i] = -(1/r_mesh) * internal_dpsi__dr_sporer(r_mesh, z_mesh, a, b, Bw, Xs, f)

        result = root(external_E_params, initial_guess, args= (1/E, Xs_arr[i], sig))
        if result.success: 
            E0, E1, E2, E3 = result.x
            e_params.append([E0, E1, E2, E3])

            dpsi__dr_ext = external_dpsi__dr(r_mesh, z_mesh, Bw, a, b, E0, E1, E2, E3)         #[T*m^2]; gradient of
            dpsi__dz_ext = external_dpsi__dz(r_mesh, z_mesh, Bw, a, b, E0, E1, E2, E3)         #external magnetic flux
            
            Br_ext_arr[i] = -(1/r_mesh) * dpsi__dz_ext            # radial magnetic field [T]
            Bz_ext_arr[i] = (1/r_mesh) * dpsi__dr_ext             # axial magnetic field [T]
        else: 
            mask.remove(i)

    a_arr       = a_arr[mask]
    b_arr       = b_arr[mask]
    Xs_arr      = Xs_arr[mask]
    elong_arr   = elong_arr[mask]
    Bw_arr      = Bw_arr[mask]

    # === finding N 
    mask = []
    for i, E in enumerate(elong_arr):
        Xs = Xs_arr[i]
        Bw = Bw_arr[i]
        a = a_arr[i]
        b = b_arr[i]
        e0, e1, e2, e3 = e_params[i]

        K = (curvature(Bw, e0, e1, e2, e3, a, 0, a, b))
        if math.isnan(K):
            pass
        else:
            N_arr.append(shape_index(a, b, K))
            mask.append(i)

    N_elong_arr = elong_arr[mask]
    N_b_arr     = b_arr[mask]
    N_a_arr     = a_arr[mask]
    N_Xs_arr    = Xs_arr[mask]
    N_Bw_arr    = Bw_arr[mask]

    N_Br_int_arr = Br_int_arr[mask]
    N_Br_ext_arr = Br_ext_arr[mask]
    N_Bz_int_arr = Bz_int_arr[mask]
    N_Bz_ext_arr = Bz_ext_arr[mask]

    # === Finding psi from N
    psi_N_arr = []
    for i, N in enumerate(N_arr):
        psi_N_arr.append(psi_stein_N(r_mesh, z_mesh, a, N_b_arr[i], N))
   
elif conII:
    print("Varying Elongation and Constant Xs")
    elong_arr   = np.linspace(1, 10, num = acc)
    Xs          = 0.6
    a           = Rc*Xs
    b_arr       = np.multiply(a, elong_arr)
    Bw          = B0/(1-(Xs**2))

    # === Finding E_n
    for i, E in enumerate(elong_arr):
        b             = b_arr[i]
        initial_guess = [Bw, Bw/3, Bw/5, 0.9]

        Br_int_arr[i] = (1/r_mesh) * internal_dpsi__dz_sporer(r_mesh, z_mesh, a, b, Bw, Xs)
        Bz_int_arr[i] = -(1/r_mesh) * internal_dpsi__dr_sporer(r_mesh, z_mesh, a, b, Bw, Xs, f)

        result = root(external_E_params, initial_guess, args= (1/E, Xs, sig))
        if result.success: 
            E0, E1, E2, E3 = result.x
            e_params.append([E0, E1, E2, E3])

            dpsi__dr_ext = external_dpsi__dr(r_mesh, z_mesh, Bw, a, b, E0, E1, E2, E3)         #[T*m^2]; gradient of
            dpsi__dz_ext = external_dpsi__dz(r_mesh, z_mesh, Bw, a, b, E0, E1, E2, E3)         #external magnetic flux
            
            Br_ext_arr[i] = -(1/r_mesh) * dpsi__dz_ext            # radial magnetic field [T]
            Bz_ext_arr[i] = (1/r_mesh) * dpsi__dr_ext             # axial magnetic field [T]
        else: 
            mask.remove(i)
            

    b_arr       = b_arr[mask]
    elong_arr   = elong_arr[mask]

    # === Finding N
    mask = []
    for i, E in enumerate(elong_arr):
        b = b_arr[i]
        e0, e1, e2, e3 = e_params[i]

        K = (curvature(Bw, e0, e1, e2, e3, a, 0, a, b))
        if math.isnan(K):
            pass
        else:
            N_arr.append(shape_index(a, b, K))
            mask.append(i)

    N_elong_arr = elong_arr[mask]
    N_b_arr = b_arr[mask]


    N_Br_int_arr = Br_int_arr[mask]
    N_Br_ext_arr = Br_ext_arr[mask]
    N_Bz_int_arr = Bz_int_arr[mask]
    N_Bz_ext_arr = Bz_ext_arr[mask]


    # === Finding psi from N
    psi_N_arr = []
    for i, N in enumerate(N_arr):
        psi_N_arr.append(psi_stein_N(r_mesh, z_mesh, a, N_b_arr[i], N))

elif conIII: 
    print("Constant Elongation and Varying Xs")
    E           = 3
    Xs_arr      = np.linspace(0.3, 0.9, num = acc)
    a_arr       = Rc*Xs_arr
    b_arr       = np.multiply(a_arr, E)
    Bw_arr      = B0/(1-(Xs_arr**2))

    for i, Xs in enumerate(Xs_arr):
        Bw = Bw_arr[i]
        a  = a_arr[i]
        b  = b_arr[i]
        initial_guess = [Bw, Bw/3, Bw/5, 0.9]

        Br_int_arr[i] = (1/r_mesh) * internal_dpsi__dz_sporer(r_mesh, z_mesh, a, b, Bw, Xs)
        Bz_int_arr[i] = -(1/r_mesh) * internal_dpsi__dr_sporer(r_mesh, z_mesh, a, b, Bw, Xs, f)

        result = root(external_E_params, initial_guess, args= (1/E, Xs, sig))
        if result.success: 
            E0, E1, E2, E3 = result.x
            e_params.append([E0, E1, E2, E3])

            dpsi__dr_ext = external_dpsi__dr(r_mesh, z_mesh, Bw, a, b, E0, E1, E2, E3)         #[T*m^2]; gradient of
            dpsi__dz_ext = external_dpsi__dz(r_mesh, z_mesh, Bw, a, b, E0, E1, E2, E3)         #external magnetic flux
            
            Br_ext_arr[i] = -(1/r_mesh) * dpsi__dz_ext            # radial magnetic field [T]
            Bz_ext_arr[i] = (1/r_mesh) * dpsi__dr_ext             # axial magnetic field [T]
        else: 
            mask.remove(i)
            

    a_arr       = a_arr[mask]
    b_arr       = b_arr[mask]
    Xs_arr      = Xs_arr[mask]
    Bw_arr      = Bw_arr[mask]

    # === Finding N
    mask = []
    for i, Xs in enumerate(Xs_arr):
        a = a_arr[i]
        b = b_arr[i]
        e0, e1, e2, e3 = e_params[i]

        K = (curvature(Bw, e0, e1, e2, e3, a, 0, a, b))
        if math.isnan(K):
            pass
        else:
            N_arr.append(shape_index(a, b, K))
            mask.append(i)

    N_b_arr = b_arr[mask]
    N_a_arr = a_arr[mask]
    N_Xs_arr = Xs_arr[mask]
    N_Bw_arr = Bw_arr[mask]


    N_Br_int_arr = Br_int_arr[mask]
    N_Br_ext_arr = Br_ext_arr[mask]
    N_Bz_int_arr = Bz_int_arr[mask]
    N_Bz_ext_arr = Bz_ext_arr[mask]


    # === Finding psi from N
    psi_N_arr = []
    for i, N in enumerate(N_arr):
        psi_N_arr.append(psi_stein_N(r_mesh, z_mesh, a, N_b_arr[i], N))

else: 
    sys.exit()


### === Plotting 
# Shape Index vs Elongation 
fig00, ax00 = plt.subplots()
ax00.scatter(N_elong_arr, N_arr, label = "Shape Index", s=4)
ax00.legend()
ax00.set_xlabel("Elongation")
ax00.set_ylabel(r"Shape Index")
fig00.savefig("output/Shape Index vs Elongation.png")

# Shape Index vs Elongation vs Normalized Separatrix Ratio
fig0, ax0 = plt.subplots()
ax0 = fig0.add_subplot(projection='3d')
ax0.scatter(N_elong_arr, N_Xs_arr, N_arr, label = "Shape Index")
ax0.legend()
ax0.set_xlabel("Elongation")
ax0.set_ylabel("Normalized Separatrix Ratio")
ax0.set_zlabel("Shape Index")
fig0.savefig("output/3D Shape Index vs Elongation vs Normalized Separatrix Ratio.png")

plt.show()
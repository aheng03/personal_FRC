# Date: 6.24.26
# Author: Adelina Hengyucius
# Shape Index Verification
'''
Understanding correlation between physical conditions and shape index 
Goal: getting a working codebase to eventually integrate into sep_geo_analysis
'''

import matplotlib.pyplot as plt #library for plotting 
from sympy import *
import numpy as np              #library for basic math functions

from func_shape_index import *
from func_steinhaurer import *


Lconv = 1e3
###---------------------------------------------------------------------------------CONSTANT PARAMETERS 
Rw                      = 6.1/Lconv        # Wall Radius [m]
Rc = Rw                 # coil radius [m]
B0 = 30                 # applied field B0 from Sporer's unreleased paper
Xs = 0.75
sig = 1.5               #flare parameter; adjustable parameter that's fixed for Steinhauer's paper
f = 1.5                 #internal psi error factor for Sporer's approximation
Bw = (B0)/(1-(Xs**2))   # Magnetic field at the midplane at the wall [T] from Steinhaur
B00 = Bw                # Sporer vacuum field [T] still working out how this relates to Steinhauer

E = 4

### === PARAMETERS 
Rs          = Xs * Rw                   # separatrix radius [m]
Zs          = E * Rs                    # separatrix half-length [m]
a = Rs
b = Zs
eps = a/b
zLen        = 150/Lconv              #liner length [m]
h           = zLen / 2              #liner height (from z=0) [m]
liner_top   = h                     #liner top [m]
liner_bot   = -h                    #liner bottom [m]

### === DOMAIN PARAMETERS
Rmax        = Rw                        #maximum r-domain value [m]
Rmin        = 0                         #minimum r-domain value [m]
Nr          = 1000                      #number of points in r-direction [\]
dr          = (Rmax - Rmin) / Nr        #radial differential [m]
Pmax        = 0.0 * np.pi/180           #maximum phi-domain value [rad] (float is in degrees)
Pmin        = 0.0 * np.pi/180           #minimum phi-domain value [rad] (float is in degrees)
Zmax        = zLen / 2                  #maximum z-domain value [m]
Zmin        = -zLen / 2                 #minimum z-domain value [m]
Nz          = 1000                      #number of points in z-direction [\]
dz          = (Zmax - Zmin) / Nz        #axial differential [m]
domx        = 1.4                       #domain multiplier to make plotting modifications easier
domy        = 0.8                       #domain multiplier to make plotting modifications easier
Rd          = Rmax * domx               #domain radius [m]
Zd          = Zmax * domy               #domain z-length [m]

r = np.arange(-Rd, Rd+dr, dr)
z = np.arange(-Zd, Zd+dz, dz)    
r_mesh, z_mesh = np.meshgrid(r, z, indexing='ij')

acc = 10               #number of variables in list -- accuracy 
###---------------------------------------------------------------------------------FUNCTIONS
def ext_flux(Xs, E0, E1, E2, A, B):
    '''
    Steinhaurer (1990) function for external flux
    '''
    for i, v in enumerate(A):
        e0 = E0[i]
        e1 = E1[i]
        e2 = E2[i]
        xs = Xs[i]

        ### === Calculated from User Input:
        R_max = xs*Rw       # equal to Rs -- separatrix radius 
        R_min = 0
        a = (R_max - R_min) / 2
        kap = 2*(B[i] / v)

        ### === DOMAIN SETUP
        domx = 1.1                                 # weight to extend the domain; helps show psi=1 curves better
        Rd = Rw
        Zd = kap * a * domx
        dr = 0.005/Lconv                           # mesh fidelity in r-dir [m]
        dz = 0.005/Lconv                           # mesh fidelity in z-dir [m]
        r = np.arange(-Rd, Rd+dr, dr)               #r-array for mesh
        z = np.arange(-Zd, Zd+dz, dz)               #z-array for mesh   
        r = np.delete(r, np.where(r==0))
        z = np.delete(z, np.where(z==0))
        R, Z = np.meshgrid(r, z, indexing='ij')     #(R,Z) mesh elements

        t1 = (Bw*v**2)/2
        t2 = e0*(R**2/v**2)
        t3 = e1*(R**2/v**2)*((R**2/v**2) - 4*(Z**2/v**2))
        t4 = e2*(((v*B[i] + Z)/((R**2 + (v*B[i] + Z)**2)**(1/2))) + ((v*B[i]-Z) / ((R**2 + (v*B[i]-Z)**2)**(1/2))))
        
        flux_val = t1 * (t2 + t3 + t4)
        flux_min = flux_val.min()

        ### === Plotting 
        # fig, ax = plt.subplots(figsize=(6, 6))
        # ax.set_aspect('equal')  # equal scaling on both axes
        # numColors = 11
        # levels = np.linspace(flux_min, 0, numColors)
        # colors = plt.cm.rainbow(np.linspace(0, 1, numColors))
        
        # cs = ax.contour(
        #     R, Z, flux_val,
        #     levels=levels,
        #     cmap='rainbow',
        #     linewidths=1.5)
        # ax.grid(alpha = 0.5)
        # fig.savefig(f"contour{i}", dpi=dpi_res)

def shape_index_N2(E0, E1, E2, alp, a, b):
    eps = a/b
    lambd = (1+(eps**2/alp**2))**(-1/2)
    num = 4*eps*E1 + 3*alp*lambd**5*E2
    den = eps**2*(-eps*E1 + alp*lambd*(2+lambd**2)*E2)
    return num/den

def shape_index_N4(E0, E1, E2, alp, a, b):
    '''
    K == curvature of separatrix at midplane (similar definition to Rs in og paper)
    '''
    eps = a/b
    lambd = (1+(eps**2/alp**2))**(-1/2)
    n_K = a*(-eps*E1 + alp*lambd*(2+lambd**2)*E2)
    d_K = 4*eps*E1 + 3*alp*lambd**5*E2
    K = n_K/d_K

    # return (K*b**2)/a
    return 1/((K*a)**2/b**4)

def internal_psi_stein_E(Es, r, z, a, b, Bw):
    
    E0, E1, E2, alpha = Es
    
    T1 = Bw*a**2/2
    T2 = E0*r**2/a**2 + E1*(r**4/a**4-4*z**2*r**2/a**4)
    T3 = E2*((alpha*b+z)/((r**2+(alpha*b+z)**2)**0.5)+(alpha*b-z)/((r**2+(alpha*b-z)**2)**0.5))
    
    return T1*(T2+T3)
#######################################################################################################
#######################################################################################################
#################################                 MAIN               ############################
#######################################################################################################
####################################################################################################### 
# Shape Index Values between 0 and 1
shape_index_lst = np.linspace(0, 1, num=acc)

#elongation (b/a) data points between 1 and 10
elongation_lst = np.linspace(1,10,num = acc)
eps_lst = 1/elongation_lst
eps_lst_copy= eps_lst.copy()

#X_s values between 0.3 and 0.9: 
Xs_lst = np.linspace(0.4,0.9,num=acc)

a_lst = Rc*Xs_lst
b_lst =  np.multiply(a_lst, elongation_lst)

e_params = []
mask = list(range(0, acc))
del_lst = []
### === E_System
for i, eps in enumerate(eps_lst_copy):
    for Xs in Xs_lst:
        #Initial guess for [E0, E1, E2, alpha] ---> Subject to change to be in terms of something else 
        initial_guess = [1.0, 0.5, 0.3, 0.2]
        
        #solve the system of equations 
        result = root(external_E_system, initial_guess, args= (eps, Xs, 1.5))
    #ensuring that correct results get added to the parameter list
    if result.success: 
        E0, E1, E2, a = result.x
        e_params.append([E0, E1, E2, a])
    else: 
        #print(f"Solution failed for eps={eps}, Xs={Xs}")
        del_lst.append(i)

for i, v in enumerate(del_lst):
    v -= i
    del mask[v]


a_lst = a_lst[mask]
b_lst = b_lst[mask]
Xs_lst = Xs_lst[mask]
elongation_lst = elongation_lst[mask]
eps_lst = eps_lst_copy[mask]

Be_arr =  B0 / (1 - Xs_lst**2)
Bw_arr = Be_arr

N2_arr = []
N4_arr = []
# for i, a in enumerate(a_lst):
    # E0 = e_params[i]['E0']
    # E1 = e_params[i]['E1']
    # E2 = e_params[i]['E2']
    # alp = e_params[i]['alpha']

    # N2_arr.append(shape_index_N2(E0, E1, E2, alp, a, b_lst[i]))
    # N4_arr.append(shape_index_N4(E0, E1, E2, alp, a, b_lst[i]))


# plt.plot(elongation_lst, N2_arr, color = 'red')
# plt.plot(elongation_lst, N1_arr, color = 'blue')
# plt.plot(elongation_lst, N4_arr, color='green')
# plt.show()


n = 1

flux_val = []
flux_val_E = []
for i, a in enumerate(a_lst):
    b = b_lst[i]
    psi = internal_psi_stein(r_mesh, z_mesh, a, b, n)
    psi_E = internal_psi_stein_E(e_params[i], r_mesh, z_mesh, a, b, Bw_arr[i])
    flux_val.append(psi)
    flux_val_E.append(psi_E)

    r, z = get_flux_contours(psi, r_mesh, z_mesh, 0, return_all = False)
    re, ze = get_flux_contours(psi_E, r_mesh, z_mesh, 0, return_all = False)

    ### === Plotting 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')  # equal scaling on both axes
    ax.plot(r, z, color=  'red', label = 'not E')
    ax.plot(re, ze, color = 'blue', label = 'E', linestyle = '-.')
    ax.grid(alpha = 0.5)
    ax.legend()
    ax.set_title(f"E={elongation_lst[i]}")
    fig.savefig(f"contour{i}", dpi=dpi_res)

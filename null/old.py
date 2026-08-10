
# Date:  5/19/26
# Shape Index Implementation 
### --- IMPORTS
import sys
import numpy as np
import matplotlib.pyplot as plt
import time 
from scipy.optimize import minimize_scalar
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.optimize import root 


from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]  # goes up from FRC to code
sys.path.append(str(base_dir))
sys.path.append("Grad_Shaf")
from _functions4plasma import *
from _plottingParameters import *
from func_flux_lifetimes import *
from func_shape_index import *
from func_steinhaurer import *
from _linerParameters import *
from _gs_im import *
start_time = time.time()
#######################################################################################################
#######################################################################################################
#################################                  MAIN                    ############################
#######################################################################################################
#######################################################################################################
acc = 5
### === INPUT PARAMETERS
T           = np.array([50, 460])                 # Temperature [eV]
B0          = 30                                  # Applied coil magnetic field [T]

#######################################################################################################
#######################################################################################################
#################################                  FUNCTIONS               ############################
#######################################################################################################
#######################################################################################################
def comparison_plots(rSep, zSep, e, a, title_txt):
    fig, ax = plt.subplots(figsize = (4, 6))
    ax.plot(rSep*Lconv, zSep*Lconv, label = r"$\boldsymbol{\psi} \boldsymbol{= 0}$", linestyle = "--", linewidth = 3)
    

    ax.set_xlabel("r [mm]", weight = 'bold')
    ax.set_ylabel("z [mm]", weight = 'bold')
    ax.set_aspect('equal')
    ax.set_xlim((-1.5*max(rSep)*Lconv, 1.5*max(rSep)*Lconv))
    ax.set_ylim((-1.3*max(zSep)*Lconv), 1.3*max(zSep)*Lconv)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    fig.savefig(title_txt)

#######################################################################################################
#######################################################################################################
#################################                 PARAMETERS               ############################
#######################################################################################################
#######################################################################################################
### === Midplane Slices
mid_r = int(len(r)/2)
mid_z = int(np.argmin(np.abs(z)))               # grabs the index of z=0
r_mask = r >= 0                                 # creates a mask to grab the r>=0 array values for lineout
### === ARRAYS
elongation_arr = np.linspace(1, 5, num = acc)
Xs_arr = np.linspace(0.3, 0.9, num = acc)      # X_s values between 0.3 and 0.9 for Shape Index
# temperature_list = np.linspace(Tstart, Tend, num = numTemps)  #eV
a_arr = Rc*Xs_arr
b_arr =  np.multiply(a_arr, elongation_arr)

Be_arr = B0 / (1-(Xs_arr)**2)
Bw_arr = Be_arr

### === INIT LISTS 
psi_int_arr         = []                       # internal pressure array
psi_ext_arr         = []                       # external pressure array
Br_ext_arr          = []
Br_int_arr          = []
Bz_ext_arr          = []
Bz_int_arr          = []
rz_og_arr           = []                        # flux contour @ separatrix w/o mask
rz_arr              = []                        # flux contour @ separatrix w/ mask

keep_lst            = []
#######################################################################################################
#######################################################################################################
#################################              IMPLEMENTATION               ###########################
#######################################################################################################
#######################################################################################################    
### === EXTERNAL & INTERNAL PRESSURE
for e_i, elong in enumerate(elongation_arr):
    Eguess = [Be_arr[e_i], Be_arr[e_i]/3, Be_arr[e_i]/5, 0.9]       # initial guess for E parameters
    psi_int_arr.append(internal_psi_sporer(R, Z, a_arr[e_i], b_arr[e_i], Bw_arr[e_i], Xs_arr[e_i], f)) 
    Br_int_arr.append((1/R) * internal_dpsi__dz_sporer(R, Z, a_arr[e_i], b_arr[e_i], Bw_arr[e_i], Xs_arr[e_i]))
    Bz_int_arr.append(-(1/R) * internal_dpsi__dr_sporer(R, Z, a_arr[e_i], b_arr[e_i], Bw_arr[e_i], Xs_arr[e_i], f))
    sol = root(                                                     # solves the system of external E equations
    fun=external_E_params,                                          # designates the function
    x0=Eguess,                                                      # initial guesses for E0, E1, E2, alpha
    args=(1/elong, Xs_arr[e_i], sig),                               # designates these parameters as constants
    method='lm',                                                    # same algorithm as old fsolve, or try 'lm'
    tol=1e-6                                                        # tolerance
    )
    if not sol.success:                                     # if E root finder is unsuccessful, display error message
        print("root() failed to converge:", sol.message)
        Br_ext = Bz_ext = 0
    else:
        keep_lst.append(e_i)
        E0, E1, E2, alpha = sol.x                           # grabs the solutions for E0, E1, E2, alpha
        dpsi__dr_ext = external_dpsi__dr(R, Z, Bw_arr[e_i], a_arr[e_i], b_arr[e_i], E0, E1, E2, alpha)         #[T*m^2]; gradient of
        dpsi__dz_ext = external_dpsi__dz(R, Z, Bw_arr[e_i], a_arr[e_i], b_arr[e_i], E0, E1, E2, alpha)         #external magnetic flux
        
        Br_ext_arr.append(-(1/R) * dpsi__dz_ext)            # radial magnetic field [T]          #[T*m]
        Bz_ext_arr.append((1/R) * dpsi__dr_ext)             # axial magnetic field [T]
        psi_ext_arr.append(external_psi(R, Z, Be_arr[e_i], a_arr[e_i], b_arr[e_i], E0, E1, E2, alpha)) 
elongation_arr           = elongation_arr[keep_lst]
Xs_arr              = Xs_arr[keep_lst]      
a_arr               = a_arr[keep_lst]
b_arr               = b_arr[keep_lst]
Be_arr              = Be_arr[keep_lst]
Bw_arr              = Be_arr
psi_int_arr         = np.array(psi_int_arr)[keep_lst]
Br_int_arr          = np.array(Br_int_arr)[keep_lst]
Bz_int_arr          = np.array(Bz_int_arr)[keep_lst]
eps_arr = 1/elongation_arr
rFit_stan_arr = 0.8*a_arr           # STANDARD
# Initializing number density & pressure arrays
numXs = len(Xs_arr)
numTemps = len(T)
Nr, Nz = R.shape
n_arr = np.zeros((numXs, Nr, Nz, numTemps))
n_avg_arr = np.zeros((numXs, Nr, Nz, numTemps))
n_max_arr = np.zeros((numXs, numTemps))
n_max_avg_arr = np.zeros((numXs, numTemps))
P_arr               = np.zeros((numXs, Nr, Nz))
P_avg_arr           = np.zeros((numXs, Nr, Nz))
### === PRESSURE
for e_i, elong in enumerate(elongation_arr):
    inside = (psi_int_arr[e_i] > 0)
    psi = np.where(inside, psi_int_arr[e_i], psi_ext_arr[e_i])    # is applied to create a full psi(r,z) [T*m^2]
    Br = np.where(inside, Br_int_arr[e_i], Br_ext_arr[e_i])       # Full radial magnetic field profile, Br(r,z) [T]
    Bz = np.where(inside, Bz_int_arr[e_i], Bz_ext_arr[e_i])       # Full axial magnetiic field profiel, Bz(r,z) [T]
    B_int = np.sqrt(Br_int_arr[e_i]**2 + Bz_int_arr[e_i]**2)
    B_ext = np.sqrt(Br_ext_arr[e_i]**2 + Bz_ext_arr[e_i]**2)
    
    Bmag = np.sqrt(Br**2 + Bz**2)                       
    ave_Bi = np.mean(B_int)
    Bmax = np.max(B_int)
    dBr__dr, dBr__dz = np.gradient(Br, r, z, edge_order=2)      # gradient of Br [T/m]
    dBz__dr, dBz__dz = np.gradient(Bz, r, z, edge_order=2)      # gradient of Bz [T/m]

    J = (1 / mu0) * (dBr__dz - dBz__dr)                         # current density [A/m^2]
    Jphi = -J
    JSlice = Jphi[r_mask, mid_z]                                # shape (Nr_pos,)

    dP__dr = Jphi * Bz                                          # partial of pressure wrt radius [Pa/m]
    dP__dz = - Jphi * Br                                        # partial of pressure wrt z [Pa/m]
    P_int = pressure_sporer(Bw_arr[e_i], Xs_arr[e_i], psi_int_arr[e_i], a_arr[e_i], b_arr[e_i], f)
    
    P = np.where(inside, P_int, 0)
    P_arr[e_i] = P
    P_avg_arr[e_i] = (np.full_like(P, np.mean(P[inside])))      # Creating a copy of P, except every value is P_avg_inside

### === NUMBER DENSITY
for Xs_i in range(numXs):
    for t_i in range(numTemps):
        temp = T[t_i] * eV/kB
        # Unaveraged Pressure Mesh 
        n_arr[:, :, :, t_i] = P_arr / (kB * temp)    
        
        # Averaged Pressure Mesh
        n_avg_arr[:, :, :, t_i] = P_avg_arr / (kB * temp)
        # Finding the max for each temperature slice at each temperature
        n_max_arr[Xs_i] = np.max(n_arr[Xs_i, :, :, t_i])      
        n_max_avg_arr[Xs_i] = np.max(n_arr[Xs_i, :, :, t_i])

n_max_arr           = np.transpose(n_max_arr)
n_max_avg_arr       = np.transpose(n_max_avg_arr)
n_max_tau_arr       = n_max_arr * (10**(-21))           # Converting to the units of the LSX number density flux equation, specific to only said equation
n_max_log_arr = n_max_arr * (10 **(-6))                 # Units from m^-3 to cm^-3

### === LIFETIMES
coulomb_arr = []
ylist_tau_clas = []
ylist_tau_LSX = []
ylist_tau_brems = []
ylist_gamma_tilt = []
for i, temp in enumerate(T):
    coulomb_arr.append(coulomb_log(temp, n_max_log_arr[i]))
    ylist_tau_clas.append(tau_clas(a_arr, temp, coulomb_arr[i]))
    ylist_tau_LSX.append(tau_LSX(a_arr, mD, temp, B0, Xs_arr))
    ylist_tau_brems.append(tau_brems(n_max_arr[i], temp))
    ylist_gamma_tilt.append(gamma_tilt(C_tilt, B0, elongation_arr, Xs_arr, n_max_arr[i], Rw))

# Instabilities
ylist_tau_tilt = tau_tilt(np.array(ylist_gamma_tilt))
# Normalized
tau_clas_norm = ylist_tau_clas /  ylist_tau_tilt
tau_LSX_norm = ylist_tau_LSX / ylist_tau_tilt
tau_brem_norm = ylist_tau_brems / (ylist_tau_tilt * 1e1)
N_arr               = np.zeros_like(elongation_arr)   # shape index

def find_rz(rSep, zSep, a):
    '''
    Getting rid of z-values less than 0
    '''
    mask = (zSep > 0) & (np.abs(rSep) < a)
    r_arr = rSep[mask]
    z_arr = zSep[mask]
    return r_arr, z_arr

# null 
def get_flux_contours(psi, R, Z, psi_level_mag, return_all=False):
    # … earlier code …

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

### === Shape Index
for e_i, elong in enumerate(elongation_arr):
    rz_og_arr.append(get_flux_contours(psi_int_arr[e_i], R, Z, 0, return_all=False))
    rz_arr.append(find_rz(rz_og_arr[e_i][0], rz_og_arr[e_i][1], a_arr[e_i]))
    eq, zs_func = poly_fit_eq(rz_arr[e_i][0], rz_arr[e_i][1], deg)
    # N_result = minimize_scalar(least_squares_fit_eps, args=(0.8*a_arr[e_i], zs_func, a_arr[e_i], elong), bounds=(1, elong), method='bounded')    
    # for i in range(len(T)):
    #     N_arr[e_i] = N_result.x
    # print(e_i)
    comparison_plots(rz_og_arr[e_i][0], rz_og_arr[e_i][1], elong, a_arr[e_i], f"Comparison Plot {e_i}")
    # print(f"index = {e_i}, E = {elong}, Xs = {Xs_arr[e_i]}, a = {a_arr[e_i]}, b = {b_arr[e_i]}, N = {N_arr[e_i]}")

#######################################################################################################
#######################################################################################################
########################                     PLOTTING                          ########################
#######################################################################################################
#######################################################################################################
temp_color_dict = {
    T[0]: 'blue',
    T[1]: 'red'
}
lt_line_dict = {
    'LSX': '-',
    'clas': '-.',
    'brem': ':'
}
# # Flux Lifetime versus Shape Index
# for t_i in range(numTemps):
#     fig, ax = plt.subplots(figsize = (figXsize, figYsize))
    
#     ax.plot(N_arr, tau_clas_norm[t_i], label = r'$\boldsymbol{\tau}_{\boldsymbol{classical}}  \ / \ \boldsymbol{\tau}_{\boldsymbol{MHD}}$', linestyle = lt_line_dict['clas'], color = temp_color_dict[T[t_i]], linewidth = lineWidth)
#     ax.plot(N_arr, tau_LSX_norm[t_i], label = r'$\boldsymbol{\tau}_{\boldsymbol{LSX}}  \ / \ \boldsymbol{\tau}_{\boldsymbol{MHD}}$', linestyle = lt_line_dict['LSX'], color = temp_color_dict[T[t_i]], linewidth = lineWidth)
#     ax.plot(N_arr, tau_brem_norm[t_i], label = r'$\boldsymbol{\tau}_{\boldsymbol{brem}}  \ / \ \boldsymbol{10\tau}_{\boldsymbol{MHD}}$', linestyle = lt_line_dict['brem'], color = temp_color_dict[T[t_i]], linewidth = lineWidth)
    
#     ax.legend(framealpha = 1, frameon = True, fontsize = legendFontSize)
#     ax.grid(alpha = 0.5)
#     ax.tick_params(axis='both', which='major', pad=Pad)
#     for label in ax.get_xticklabels() + ax.get_yticklabels():
#         label.set_fontweight('bold')
    
#     ax.set_xlabel("Shape Index", weight = 'bold', fontsize = textFontSize)
#     ax.set_ylabel("Flux Lifetime", weight = 'bold', fontsize = textFontSize)
#     fig.tight_layout()
#     fig.savefig(f"{T[t_i]} Flux Lifetime vs Shape Index")

# Flux Lifetimes versus Elongations 
# for t_i in range(numTemps):
#     fig2, ax2 = plt.subplots(figsize = (figXsize, figYsize))
#     ax2.plot(elongation_arr, tau_clas_norm[t_i], label = r'$\boldsymbol{\tau}_{\boldsymbol{classical}}  \ / \ \boldsymbol{\tau}_{\boldsymbol{MHD}}$', linestyle = lt_line_dict['clas'], color = temp_color_dict[T[t_i]], linewidth = lineWidth)
#     ax2.plot(elongation_arr, tau_LSX_norm[t_i], label = r'$\boldsymbol{\tau}_{\boldsymbol{LSX}}  \ / \ \boldsymbol{\tau}_{\boldsymbol{MHD}}$', linestyle = lt_line_dict['LSX'], color = temp_color_dict[T[t_i]], linewidth = lineWidth)
#     ax2.plot(elongation_arr, tau_brem_norm[t_i], label = r'$\boldsymbol{\tau}_{\boldsymbol{brem}}   \ / \ \boldsymbol{10\tau}_{\boldsymbol{MHD}}$', linestyle = lt_line_dict['brem'], color = temp_color_dict[T[t_i]], linewidth = lineWidth)
#     ax2.legend(framealpha = 1, frameon = True, fontsize = legendFontSize)
#     ax2.grid(alpha = 0.5)
#     ax2.tick_params(axis='both', which='major', pad=Pad)
#     for label in ax2.get_xticklabels() + ax2.get_yticklabels():
#         label.set_fontweight('bold')
#     ax2.set_xlabel("Elongation", weight = 'bold', fontsize = textFontSize)
#     ax2.set_ylabel("Flux Lifetime", weight = 'bold', fontsize = textFontSize)
#     fig2.tight_layout()
#     fig2.savefig(f"{T[t_i]} Flux Lifetime vs Elongation")

# # Shape Index versus Elongations 
# # fig3, ax3 = plt.subplots(figsize = (figXsize, figYsize))
# # ax3.plot(elong_arr, N_arr, color = 'blue', linewidth = lineWidth)
# # ax3.grid(alpha = 0.5)
# # ax3.tick_params(axis='both', which='major', pad=Pad)
# # for label in ax3.get_xticklabels() + ax3.get_yticklabels():
# #     label.set_fontweight('bold')
# # ax3.set_xlabel("Elongation", weight = 'bold', fontsize = textFontSize)
# # ax3.set_ylabel("Shape Index", weight = 'bold', fontsize = textFontSize)
# # fig3.savefig("Elongation vs Shape Index")
# print("--- %s seconds ---" % (time.time() - start_time))

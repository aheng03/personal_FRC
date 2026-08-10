#######################################################################################################
#######################################################################################################
#################################                  IMPORTS                 ############################
#######################################################################################################
#######################################################################################################
import matplotlib.pyplot as plt #library for plotting 
import numpy as np              #library for basic math functions 
import sys 
from pathlib import Path 

from _functions4plasma import * 
from _plottingParameters import *
from func_steinhaurer import *
from func_flux_lifetimes import *
from _linerParameters import *

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.signal import find_peaks

#######################################################################################################
#######################################################################################################
#################################                  MAIN                    ############################
#######################################################################################################
#######################################################################################################

Lconv = 1e-2
model = "sporer"
###---------------------------------------------------------------------------------CONSTANT PARAMETERS 
sig = 1.5               # flare parameter; adjustable parameter that's fixed for Steinhauer's paper
f = 1.5                 # internal psi error factor for Sporer's approximation
Rw = 0.61*Lconv         # wall radius [m]
Rc = Rw                 # coil radius [m]

### === USER INPUT
T = np.array([50, 460]) #eV
E = np.array([4.0, 2.9])
B0 = 30                 # applied field B0 from Sporer's unreleased paper 
# n_array_max_unavg = np.array([6.55e25, 1.33e25])


Lx = 3.00*Lconv         # FRC length [m]
Z0 = Lx / 2             # FRC half-length [m]



acc = 5              #number of variables in list -- accuracy
###---------------------------------------------------------------------------------FUNCTIONS
# --- Finding the peaks 
def find_x_peak(x_list, y_list, y_list_peak):
    for index, value in enumerate(x_list): 
        if y_list[index] == y_list_peak:
            return value

#######################################################################################################
#######################################################################################################
#################################                 PARAMETERS               ############################
#######################################################################################################
#######################################################################################################     
### === Midplane Slices
mid_r = int(len(r)/2)
mid_z = int(np.argmin(np.abs(z)))               # grabs the index of z=0
r_mask = r >= 0   

###--------------------------------------------------------------------------------------GRAPHING PARAMETERS    
elongation_arr = np.linspace(1,10,num= acc)
eps_arr = 1/elongation_arr
Xs_arr = np.linspace(0.3, 0.9, num=acc)

Be_arr = (B0)/(1-(Xs_arr**2))   # Magnetic field at the midplane at the wall [T] from Steinhaur
Bw_arr = Be_arr                # Sporer vacuum field [T] still working out how this relates to Steinhauer

a_arr = Rc*Xs_arr
b_arr = np.multiply(Xs_arr, elongation_arr)
b_dict = {}
a_dict = {}

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


elongation_arr      = elongation_arr[keep_lst]
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
n_arr = np.zeros((numTemps, numXs, Nr, Nz))
n_avg_arr = np.zeros((numTemps, numXs, Nr, Nz))
n_max_arr = np.zeros((numTemps, numXs))
n_max_avg_arr = np.zeros((numTemps, numXs))

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
        n_arr[t_i, Xs_i, :, : ] = P_arr[Xs_i] / (kB * temp)    
        
        # Averaged Pressure Mesh
        n_avg_arr[t_i, Xs_i, :, : ] = P_avg_arr[Xs_i] / (kB * temp)

        # Finding the max for each temperature slice at each temperature
        n_max_arr[t_i, Xs_i] = np.max(n_arr[t_i, Xs_i, :, : ])
        n_max_avg_arr[t_i, Xs_i] = np.max(n_arr[t_i, Xs_i, :, : ])


n_max_tau_arr       = n_max_arr * (10**(-21))           # Converting to the units of the LSX number density flux equation, specific to only said equation
n_max_unavg_log_arr = n_max_arr * (10 **(-6))           # Units from m^-3 to cm^-3


### === Initializing arrays
coulomb_arr = []
ylist_tau_clas = {}
ylist_tau_LSX = {}
ylist_tau_brems = {}
ylist_gamma_tilt = {}
ylist_tau_tilt = {}


### === COULOMB VALUES
for t_i, temp in enumerate(T):
    coulomb_arr.append(coulomb_log(temp, n_max_unavg_log_arr[t_i]))

### === Flux Lifetimes
for idx, temp in enumerate(T): 
    ylist_tau_clas[idx] = tau_clas(a_arr, temp, coulomb_arr[idx])
    ylist_tau_LSX[idx] = tau_LSX(a_arr, mD, temp, B0, Rw)   
    ylist_tau_brems[idx] = tau_brems(n_max_arr[idx], temp)

    ylist_gamma_tilt.setdefault(idx, []) 
    ylist_tau_tilt.setdefault(idx, [])
    gamma_tilt_value = gamma_tilt(C_tilt, B0, elongation_arr, Xs_arr, n_array_max[idx], Rw)
    ylist_gamma_tilt[idx].append(gamma_tilt_value)
    ylist_tau_tilt[idx].append(tau_tilt(gamma_tilt_value))


# # taking it just for each temperature value 
# for idx in range(len(temperature_list)):
#     T = temperature_list[idx]
#     ylist_tau_brems_norm.setdefault(idx, [])
#     for value in range(len(elongation_list)):
#         ylist_tau_brems_norm[idx].append(ylist_tau_brems[idx] / (100*ylist_tau_tilt[idx][value]))
 
# for idx in range(len(temperature_list)):
#     T = temperature_list[idx]
#     ylist_tau_clas_norm.setdefault(idx, []) 
#     ylist_tau_LSX_norm.setdefault(idx, []) 
#     ylist_gamma_tilt.setdefault(idx, [])
#     for value in range(len(elongation_list)):
#         ylist_tau_clas_norm[idx].append(ylist_tau_clas[idx][value] / ylist_tau_tilt[idx][value])
#         ylist_tau_LSX_norm[idx].append(ylist_tau_LSX[idx][value] / ylist_tau_tilt[idx][value])
#         # ylist_gamma_tilt[idx].append(gamma_tilt(elongation_list[value], Xs[idx], Rw, n_array_max_unavg[idx]))

# #######################################################################################################
# ####################################################################################################### 
# #################################           COMPARISON PLOTS               ############################
# #######################################################################################################
# #######################################################################################################
# plot_num = 0            #indexing for plot #

# colors = {
#     'clas': "C0",        # first color in the cycle
#     'LSX': "C1",         # second color
#     'brem': "C2",        # third color
# }

# # 0 - HOT, 1 - OPT
# lstyles = {
#     0: 'dashed',
#     1: 'solid',
# }

# ### --- Normalized Confinement Lifetime Graph 
# '''
# peaks_clas = max(ylist_tau_clas_norm[0])
# peak_clas_elong = find_x_peak(elongation_list, ylist_tau_clas_norm[0], peaks_clas)
# valley_clas = min(ylist_tau_clas_norm[0])
# valley__clas_elong = find_x_peak(elongation_list, ylist_tau_clas_norm[0], valley_clas)

# peaks_LSX = max(ylist_tau_LSX_norm[0])
# peak_LSX_elong = find_x_peak(elongation_list, ylist_tau_LSX_norm[0], peaks_LSX)
# valley_LSX = min(ylist_tau_LSX_norm[0])
# valley__LSX_elong = find_x_peak(elongation_list, ylist_tau_LSX_norm[0], valley_LSX)
# '''

# fig, ax1 = plt.subplots(num=plot_num, dpi=dpi_res, figsize=(figXsize, figYsize))      # width=9in, height=6in
# ax1.set_xlim(1, 10)
# ax1.set_ylim(0,24)

# ax1.spines['bottom'].set_position(('data', 0))
# # Primary axis (tau ratios)
# for idx, T in enumerate(temperature_list):
#     ax1.plot(elongation_list, ylist_tau_brems_norm[idx], color=colors['brem'], label=r'$\tau_{brems} \ / \ 100 \tau_{MHD}$', linewidth = lineWidth, linestyle = lstyles[idx])
#     ax1.plot(elongation_list, ylist_tau_clas_norm[idx], color=colors['clas'], label=r'$\tau_{class} \ / \ \tau_{MHD}$', linewidth=lineWidth, linestyle = lstyles[idx] )
#     ax1.plot(elongation_list, ylist_tau_LSX_norm[idx], color=colors['LSX'], label=r'$\tau_{LSX} \ / \ \tau_{MHD}$', linewidth=lineWidth, linestyle = lstyles[idx])

# # ---- integer ticks for grid (MAJOR) ----
# ax1.ticklabel_format(axis='y', style='plain', scilimits=(0,0))
# x_ticks = np.arange(1, 11, 1)
# ax1.set_xticks(x_ticks)
# ax1.set_xticklabels(x_ticks,
#                     weight='bold', fontsize=textFontSize)

# # ---- grid only on integers ----
# ax1.grid(which='major')

# # ---- labels ----
# ax1.set_xlabel("Elongation", fontsize=labelFontSize, weight='bold')
# ax1.set_ylabel(r'$\tau_{\Phi} \ / \ \tau_{MHD}$',
#                fontsize=labelFontSize, weight='bold')

# # ---- y ticks ----
# y_ticks = np.arange(0, 26, 2)
# ax1.set_yticks(y_ticks)
# ax1.set_yticklabels(y_ticks,
#                     weight='bold', fontsize=textFontSize)


# #lines_1, labels_1 = ax1.get_legend_handles_labels() # Combine legends

# # color path legends
# blue_patch = mpatches.Patch(color='C0', label = r'$\tau_{class} \ / \ \tau_{MHD}$')
# orange_patch = mpatches.Patch(color="C1", label = r'$\tau_{LSX} \ / \ \tau_{MHD}$')
# green_patch = mpatches.Patch(color="C2", label = r'$\tau_{brems} \ / \ 100 \tau_{MHD}$')

# # Combine legends
# legend_values = [blue_patch, orange_patch, green_patch,
#                  Line2D([0], [0], color='black', linestyle ='solid', label = 'Optimized'), 
#                 Line2D([0], [0], color = 'black', linestyle='dashed', label = 'Hot')]
# # Place legend outside the plot area
# ax1.legend(handles=legend_values,
#            fontsize = legendFontSize,
#            loc='upper right',
#            frameon=True,
#            facecolor='white',
#            edgecolor='black',
#            framealpha=1)

# title_save = f"normalized con lifetimes vs elongation"
# plt.savefig(title_save + ".png", dpi=dpi_res, bbox_inches='tight')
# plot_num += 1
'''
# PLOT 2
fig, ax2 = plt.subplots(num=plot_num, dpi=dpi_res, figsize=(8, 3))      # width=9in, height=3in
# Secondary axis (gamma_tilt / gamma_MHD)
for T in temperature_list:
    ax2.plot(elongation_list, ylist_gamma_tilt_MHD[T], color=colors[T], label= str(T) + r'$\gamma_{tilt} \ / \ \gamma_{MHD}$', linewidth=2, linestyle='dashed')
ax2.set_xlabel("Elongation", fontsize=labelFontSize, weight='bold')
ax2.grid()
ax2.set_xlim(1, 10)
ax2.legend()
y2_ticks = np.arange(0, 1.1, 0.1)
ax2.set_xticks(x_ticks)
ax2.set_yticks(y2_ticks)
ax2.spines['bottom'].set_position(('data', 0))
ax2.set_ylabel(r'$\gamma_{tilt}  \ / \ \gamma_{MHD}$', fontsize=labelFontSize, labelpad=10, weight='bold')
ax2.tick_params(axis='y', labelcolor='black')
ax2.ticklabel_format(axis='y', style='plain', scilimits=(0,0))
ax2.set_xticklabels(ax1.get_xticks(), weight = 'bold')
ax2.set_yticklabels(ax2.get_yticks(), weight = 'bold')
ax2.tick_params(axis='x', which='major')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Adjust spacing to avoid label overlap
fig.subplots_adjust(right=0.82)  # moves the right side of the plot inward a bit

title_save = f"normalized tilt vs elongation, T = {int(temperature_list[0])} eV"
plt.savefig(title_save + ".png", dpi=dpi_res, bbox_inches='tight')
plot_num += 1


print(f"maximuim classical confinement value: ({peak_clas_elong}, {peaks_clas})" )
print(f"maximum LSX confinement value:({peak_LSX_elong}, {peaks_LSX})" )
print(f"minimum classical confinement value: ({valley__clas_elong}, {valley_clas})" )
print(f"minimum LSX confinement value:({valley__LSX_elong}, {valley_LSX})" )
'''

print("Inputs: \n" \
"B0:", B0, "T \n", \
"T:", T, "in eV \n", \
# "n0:", n_array_max_unavg, "in m^-3 \n",\
"Xs:", Xs_arr, "\n",\
"Rw:", Rw, "m \n", \
"E:", E)

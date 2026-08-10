# Author: Adelina Hengyucius
# Date:  3/28/26
# Shape Index Implementation 
'''
based off the most recent shape index paper where 1<N<E
'''
### --- IMPORTS
import sys
from scipy.optimize import minimize_scalar

from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]  # goes up from FRC to code
sys.path.append(str(base_dir))

from _functions4plasma import *
from _plottingParameters import * 
from _linerParameters import *
from func_shape_index import *
from func_steinhaurer import *


#######################################################################################################
#######################################################################################################
#################################                  MAIN                    ############################
#######################################################################################################
#######################################################################################################
E           = 5.0                       # Elongation
Xs          = 0.6                       # normalized separatrix radius [\]
T           = 50                        # Temperature [eV]
B0          = 30                        # applied coil magnetic field [T]

zLen        = 150/Lconv                        #liner length [mm]
Rw          = 6.1/Lconv                       #flux-conserving wall radius [mm]

Rs          = Xs * Rw                   #separatrix radius [mm]
Be          = B0 / (1 - Xs**2)          #external midplane magnetic field [T]
Bw          = Be                        #wall magnetic field [T]

a = Rs
b = E*Rs
eps = 1/E

# Domain Parameters
Zmax        = zLen / 2                  #maximum z-domain value [mm]
Zmin        = -zLen / 2                 #minimum z-domain value [mm]
Rmax        = Rw                        #maximum r-domain value [mm]
Rmin        = 0                         #minimum r-domain value [mm]
Rd          = Rmax * domx               #domain radius [mm]
Zd          = Zmax * domx               #domain z-length [mm]

sig=1.5
f=1.5
r_fit_stan = 0.8*a

deg = 20

### === DOMAIN SETUP
psi_int = internal_psi_sporer(R, Z, a, b, Bw, Xs, f)
rz_og_arr = get_flux_contours(psi_int, R, Z, 0, return_all= False)

rz_arr = find_rz(rz_og_arr[0], rz_og_arr[1], a)
eq, zs_func = poly_fit_eq(rz_arr[0], rz_arr[1], deg)

# # Arrays 
# percentage_arr = np.linspace(0, 0.01, 100)
# r_fit_arr = a*percentage_arr

#######################################################################################################
#######################################################################################################
########################                     IMPLEMENTATION                    ########################
#######################################################################################################
#######################################################################################################

### === Finding convergence for best r_fit value
# N_con_array = []
# for rf in r_fit_arr:
#     N_result = minimize_scalar(least_squares_fit_eps, args=(r_fit_stan, zs_func, a, E), bounds=(1, E), method='bounded')
#     N_con_array.append(N_result.x)

### === Finding N for r_fit_stan
N_result = minimize_scalar(least_squares_fit_eps, args=(r_fit_stan, zs_func, a, E), bounds=(1, E), method='bounded')
N = N_result.x

z_s_arr = zs_func(rz_arr[0])
z_cap_arr = z_cap(rz_arr[0], N, a, E)

error_mar_zs = []
error_mar_cap = []
### === Error Margins
for z_i, z_v in enumerate(rz_arr[1]):
    error_mar_zs.append(((z_v) - z_s_arr[z_i]))
    error_mar_cap.append(((z_v) - z_cap_arr[z_i]))

    

#######################################################################################################
#######################################################################################################
########################                     PLOTTING                          ########################
#######################################################################################################
#######################################################################################################
labelFontWeight = 'bold'
figXsize = 8
figYsize = 5
legendFontSize = 15
lineWidth = 4
tickFontSize = 15

# R_convergence Plot
# fig1, ax1 = plt.subplots(dpi=150)
# ax1.plot(percentage_arr, N_con_array)
# ax1.set_title("N_value vs R_fit")
# ax1.set_xlabel("Percentage of a")
# ax1.set_ylabel("N")


# Comparison Plot [Vertical]
fig2, ax2 = plt.subplots(figsize=(figXsize, figYsize))
ax2.plot(rz_og_arr[0]*Lconv, rz_og_arr[1]*Lconv, color="blue", linestyle = ":", label = r"$\boldsymbol{\psi = 0}$", linewidth =lineWidth)
ax2.plot(rz_arr[0]*Lconv, z_s_arr*Lconv, color="orange", label = r"$\boldsymbol{Z}$", linestyle = '-.', linewidth =lineWidth)           # standardized separatrix shape
ax2.plot(rz_arr[0]*Lconv, z_cap_arr*Lconv, color = "red", linestyle = "-", label = r"$\boldsymbol{Z}_{\boldsymbol{cap}}$", linewidth =lineWidth)

ax2.annotate("", xy = (Rs*Lconv*1.05,4), xytext = (0, 4), arrowprops = dict(arrowstyle="->", lw = 1, color = 'blue') )
ax2.annotate(r"$\boldsymbol{a}$", xy = ((Rs*Lconv)/2, 3), fontsize = textFontSize)
ax2.annotate("", xy = (0,b*Lconv*1.02), xytext = (0, 0), arrowprops = dict(arrowstyle="->", lw = 1, color = 'blue') )
ax2.annotate(r"$\boldsymbol{b}$", xy = (-1,(b*Lconv)-4), fontsize = textFontSize)
ax2.annotate("", xy = (Rs*Lconv*0.8*1.05,6), xytext = (0, 6), arrowprops = dict(arrowstyle="->", lw = 1, color = 'blue') )
ax2.annotate(r"$\boldsymbol{r}_{\boldsymbol{f}}$", xy = ((Rs*Lconv)/2, 5), fontsize = textFontSize)

ax2.set_aspect('equal')
ax2.legend(loc='center left', bbox_to_anchor=(1.1, 0.5),
            frameon=False, borderaxespad=0.0, fontsize=legendFontSize)
ax2.set_xlim(-Rd*0.85*Lconv, Lconv*Rd*0.85)
ax2.set_ylim(0, Zd*0.25*Lconv)
ax2.set_xlabel(r"$\boldsymbol{r}$  $\boldsymbol{[mm]}$", fontsize = textFontSize)
ax2.set_ylabel(r"$\boldsymbol{z}$  $\boldsymbol{[mm]}$", fontsize = textFontSize)

ax2.tick_params(axis='both', labelsize=12)
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight('bold')
fig2.savefig("Comparison Plot_v")

# Comparison Plot [Horizontal]
# fig3, ax3 =  plt.subplots(figsize=(12, 8))
# ax3.plot(rz_og_arr[1], rz_og_arr[0], color="green", label = r"Flux $\boldsymbol{\psi = 0}$", linewidth =lineWidth)
# ax3.plot(z_s_arr, rz_arr[0], color="red", linestyle = "-.", label = r"$\boldsymbol{Z}_{\boldsymbol{s}}$", linewidth =lineWidth)           # standardized separatrix shape
# ax3.plot(z_cap_arr, rz_arr[0], color = "blue", linestyle = ":", label = r"$\boldsymbol{Z}_{\boldsymbol{cap}}$", linewidth =lineWidth)
# ax3.legend(loc='lower right', bbox_to_anchor=(0.25, 0.1),
#             frameon=True, borderaxespad=0.0, prop ={'weight': 'bold', 'size': legendFontSize})
# ax3.set_ylim(0, Rd*0.85)
# ax3.set_xlim(0, Zd*0.6)
# ax3.text(-5, 60, 'Parabola $Y = x^2$', fontsize=15)
# ax3.set_xlabel(r"$\boldsymbol{z}$  $\boldsymbol{[mm]}$", size  = tickFontSize, labelpad = 10)
# ax3.set_ylabel(r"$\boldsymbol{r}$  $\boldsymbol{[mm]}$", size = tickFontSize, labelpad = 10)
# ax3.tick_params(axis='both', labelsize=tickFontSize)
# for label in ax3.get_xticklabels() + ax3.get_yticklabels():
#     label.set_fontweight('bold')
# ax3.set_aspect('equal', adjustable='box')
# ax3.grid(alpha = 0.8)
# fig3.savefig("shape_index")

# Error Plot
# fig3, ax3 = plt.subplots(dpi=150)
# ax3.scatter(rz_arr[0], error_mar_zs, color='blue', s=2, label = "zs")
# ax3.scatter(rz_arr[0], error_mar_cap, color='orange', s=2, label = "cap")
# ax3.legend()
# ax3.set_title("Error")


# Print Statements 
print("best_N:", N)
print("Elongation:", E)
print("a", a)
#######################################################################################################
#######################################################################################################
#################################                  IMPORTS                 ############################
#######################################################################################################
#######################################################################################################
import matplotlib.pyplot as plt #library for plotting 
import numpy as np              #library for basic math functions 

from _plottingParameters import *  
from _functions4plasma import*
from func_flux_lifetimes import *
from func_steinhaurer import *

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from scipy.optimize import fsolve
from scipy.optimize import root 
from mpl_toolkits.mplot3d import Axes3D                         #registers the 3D projection 
from matplotlib.ticker import AutoMinorLocator
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter
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
Xs = np.array([0.89, 0.73])               # normalized separatrix ratio [\]
temperature_list = np.array([460, 460]) #eV
n_array_max_unavg = np.array([6.55e25, 1.33e25])
E = np.array([2.9, 4.5])
B0 = 30                 # applied field B0 from Sporer's unreleased paper 


eps = 1/E             # inverse elongation
Rs = Xs * Rw            # separatrix radius [m]
Lx = 3.00*Lconv         # FRC length [m]
Z0 = Lx / 2             # FRC half-length [m]
Bw = (B0)/(1-(Xs**2))   # Magnetic field at the midplane at the wall [T] from Steinhaur
B00 = Bw                # Sporer vacuum field [T] still working out how this relates to Steinhauer
a = Rs                  # FRC semi-minor axis [m]
b = E*a                  # FRC semi-major axis [m] 

###----------------------------------------------------------------------------------------DOMAIN SETUP
q = ee

Nu_normal = 1.03* 10**(-4)            # classical cross-field Spitzer resistivity (N_normal_clas) from Sporer's paper about flux lifetimes 
A_brems = 1.6*10**(-38)     # from Sporer's paper about flux lifetimes [Wm^3 / sqrt(eV)]
C_tilt = 1              # ranges from 1 to 2, just set to 1 for now 
Cth = 1                 # empirical fittings for chodura
f_nu = 1                   # empirical fittings for chodura
Z_spitzer = 1

h = 4.00*Lconv         #length of liner [m]
domx = 1.1                                  #weight to extend the domain; helps show psi=1 curves better
Rd = Rw * domx                              #half-length of computational domain in r-dir [m]
Zd = h/2 * domx                             #half-length of computational domain in z-dir [m]

dr = 0.005*Lconv                           #mesh fidelity in r-dir [m]
dz = 0.005*Lconv                           #mesh fidelity in z-dir [m]
r = np.arange(-Rd, Rd+dr, dr)               #r-array for mesh
z = np.arange(-Zd, Zd+dz, dz)               #z-array for mesh
R, Z = np.meshgrid(r, z, indexing='ij')     #(R,Z) mesh elements

acc = 1000              #number of variables in list -- accuracy
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

###--------------------------------------------------------------------------------------NUMBER DENSITY
Nr, Nz = R.shape

n_array_max_log_Punavg = []
for i in n_array_max_unavg:
    n_array_max_log_Punavg.append( i * 10 **(-6))     # Units from m^-3 to cm^-3

#Coulomb values for corresponding T values
coulomb_list = []
for index, temp in enumerate(temperature_list): 
    coulomb_list.append(coulomb_log(temp, n_array_max_log_Punavg[index]))
    
###--------------------------------------------------------------------------------------GRAPHING PARAMETERS    

#elongation (b/a) data points between 1 and 10
elongation_list = np.linspace(1,10,num= acc)

#inverse elongation (a/b)
eps_list = 1/elongation_list

b_dict = {}
a_dict = {}
for idx, t in enumerate(temperature_list):
    b_dict[idx] = np.linspace(0.0001, b[idx], num = acc)
    a_dict[idx] = b_dict[idx] * eps_list

#y_lists for tau_clas for every temperature value 
ylist_tau_clas = {}
for idx, temp in enumerate(temperature_list): 
    ylist_tau_clas[idx] = tau_clas(a_dict[idx], temp, coulomb_list[idx])

#y_lists for tau_LSX for every temperature value 
ylist_tau_LSX = {}
for idx, temp in enumerate(temperature_list):
    ylist_tau_LSX[idx] = tau_LSX(a_dict[idx], mD, temp, B0, Rw)    
    
#y_list for tau_num & tau_brems 
ylist_tau_brems = {}            # Bremsstrahlung radiation ylist 
for idx, temp in enumerate(temperature_list):
    ylist_tau_brems[idx] = tau_brems(n_array_max_unavg[idx], temp)
###--------------------------------------------------------------------------------------GRAPHING PARAMETERS (10/25) 
# tilt lifetimes
ylist_gamma_MHD = {}
ylist_gamma_tilt = {} 
ylist_tau_tilt = {}
ylist_tau_MHD = {}

for index, value in enumerate(n_array_max_unavg):
    T = temperature_list[index]
    ylist_gamma_MHD.setdefault(index, []) 
    ylist_gamma_tilt.setdefault(index, []) 
    ylist_tau_tilt.setdefault(index, [])
    ylist_tau_MHD.setdefault(index, [])
    for e_index, e_value in enumerate(elongation_list):
        gamma_MHD_value = (gamma_MHD(C_tilt, B0, e_value, Xs[index], value, Rw))
        ylist_gamma_MHD[index].append(gamma_MHD_value)
        
        ylist_tau_MHD[index].append(tau_MHD(C_tilt, B0, e_value, Xs[index], value, Rw)) 
        
        gamma_tilt_value = gamma_tilt(gamma_MHD_value, temperature_list[index], B0, e_value, Xs[index], Rw)
        ylist_gamma_tilt[index].append(gamma_tilt_value)

        ylist_tau_tilt[index].append(tau_tilt(gamma_tilt_value))

### --- normalized list init 
ylist_tau_clas_norm = {}
ylist_tau_LSX_norm = {}
ylist_gamma_tilt_MHD = {}
ylist_tau_brems_norm = {}

# taking it just for each temperature value 
for idx in range(len(temperature_list)):
    T = temperature_list[idx]
    ylist_tau_brems_norm.setdefault(idx, [])
    for value in range(len(elongation_list)):
        ylist_tau_brems_norm[idx].append(ylist_tau_brems[idx] / (100*ylist_tau_MHD[idx][value]))
 
for idx in range(len(temperature_list)):
    T = temperature_list[idx]
    ylist_tau_clas_norm.setdefault(idx, []) 
    ylist_tau_LSX_norm.setdefault(idx, []) 
    ylist_gamma_tilt_MHD.setdefault(idx, [])
    for value in range(len(elongation_list)):
        ylist_tau_clas_norm[idx].append(ylist_tau_clas[idx][value] / ylist_tau_MHD[idx][value])
        ylist_tau_LSX_norm[idx].append(ylist_tau_LSX[idx][value] / ylist_tau_MHD[idx][value])
        ylist_gamma_tilt_MHD[idx].append(gamma_tilt_MHD(elongation_list[value], Xs[idx], Rw, n_array_max_unavg[idx]))

#######################################################################################################
####################################################################################################### 
#################################           COMPARISON PLOTS               ############################
#######################################################################################################
#######################################################################################################
plot_num = 0            #indexing for plot #

colors = {
    'clas': "C0",        # first color in the cycle
    'LSX': "C1",         # second color
    'brem': "C2",        # third color
}

# 0 - HOT, 1 - OPT
lstyles = {
    0: 'dashed',
    1: 'solid',
}

### --- Normalized Confinement Lifetime Graph 
'''
peaks_clas = max(ylist_tau_clas_norm[0])
peak_clas_elong = find_x_peak(elongation_list, ylist_tau_clas_norm[0], peaks_clas)
valley_clas = min(ylist_tau_clas_norm[0])
valley__clas_elong = find_x_peak(elongation_list, ylist_tau_clas_norm[0], valley_clas)

peaks_LSX = max(ylist_tau_LSX_norm[0])
peak_LSX_elong = find_x_peak(elongation_list, ylist_tau_LSX_norm[0], peaks_LSX)
valley_LSX = min(ylist_tau_LSX_norm[0])
valley__LSX_elong = find_x_peak(elongation_list, ylist_tau_LSX_norm[0], valley_LSX)
'''

fig, ax1 = plt.subplots(num=plot_num, dpi=dpi_res, figsize=(figXsize, figYsize))      # width=9in, height=6in
ax1.set_xlim(1, 10)
ax1.set_ylim(0,24)

ax1.spines['bottom'].set_position(('data', 0))
# Primary axis (tau ratios)
for idx, T in enumerate(temperature_list):
    ax1.plot(elongation_list, ylist_tau_brems_norm[idx], color=colors['brem'], label=r'$\tau_{brems} \ / \ 100 \tau_{MHD}$', linewidth = lineWidth, linestyle = lstyles[idx])
    ax1.plot(elongation_list, ylist_tau_clas_norm[idx], color=colors['clas'], label=r'$\tau_{class} \ / \ \tau_{MHD}$', linewidth=lineWidth, linestyle = lstyles[idx] )
    ax1.plot(elongation_list, ylist_tau_LSX_norm[idx], color=colors['LSX'], label=r'$\tau_{LSX} \ / \ \tau_{MHD}$', linewidth=lineWidth, linestyle = lstyles[idx])

# ---- integer ticks for grid (MAJOR) ----
ax1.ticklabel_format(axis='y', style='plain', scilimits=(0,0))
x_ticks = np.arange(1, 11, 1)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_ticks,
                    weight='bold', fontsize=textFontSize)

# ---- grid only on integers ----
ax1.grid(which='major')

# ---- labels ----
ax1.set_xlabel("Elongation", fontsize=labelFontSize, weight='bold')
ax1.set_ylabel(r'$\tau_{\Phi} \ / \ \tau_{MHD}$',
               fontsize=labelFontSize, weight='bold')

# ---- y ticks ----
y_ticks = np.arange(0, 26, 2)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_ticks,
                    weight='bold', fontsize=textFontSize)


#lines_1, labels_1 = ax1.get_legend_handles_labels() # Combine legends

# color path legends
blue_patch = mpatches.Patch(color='C0', label = r'$\tau_{class} \ / \ \tau_{MHD}$')
orange_patch = mpatches.Patch(color="C1", label = r'$\tau_{LSX} \ / \ \tau_{MHD}$')
green_patch = mpatches.Patch(color="C2", label = r'$\tau_{brems} \ / \ 100 \tau_{MHD}$')

# Combine legends
legend_values = [blue_patch, orange_patch, green_patch,
                 Line2D([0], [0], color='black', linestyle ='solid', label = 'Optimized'), 
                Line2D([0], [0], color = 'black', linestyle='dashed', label = 'Hot')]
# Place legend outside the plot area
ax1.legend(handles=legend_values,
           fontsize = legendFontSize,
           loc='upper right',
           frameon=True,
           facecolor='white',
           edgecolor='black',
           framealpha=1)

title_save = f"normalized con lifetimes vs elongation"
plt.savefig(title_save + ".png", dpi=dpi_res, bbox_inches='tight')
plot_num += 1
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
"T:", temperature_list, "in eV \n", \
"n0:", n_array_max_unavg, "in m^-3 \n",\
"Xs:", Xs, "\n",\
"Rw:", Rw, "m \n", \
"E:", E)

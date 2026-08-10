import numpy as np 
import matplotlib.pyplot as plt #library for plotting 
from matplotlib.animation import FuncAnimation
from func_steinhaurer import *
from _plottingParameters import *

Lconv = 1e3
acc = 20
###---------------------------------------------------------------------------------CONSTANT PARAMETERS 
Rw = 6.1/Lconv          # Wall Radius [m]
B0 = 30                 # applied field B0 from Sporer's unreleased paper
Xs = 0.75
E = 4
zLen = 150/Lconv        #liner length [m]

### === PARAMETERS
Bw = (B0)/(1-(Xs**2))   # Magnetic field at the midplane at the wall [T] from Steinhaur
B00 = Bw                # Sporer vacuum field [T] still working out how this relates to Steinhauer
Rc = Rw                 # coil radius [m]
Rs = Xs * Rw            # separatrix radius [m]
Zs = E * Rs             # separatrix half-length [m]
a = Rs
b = Zs
eps = a/b

sig = 1.5               #flare parameter; adjustable parameter that's fixed for Steinhauer's paper
f = 1.5                 #internal psi error factor for Sporer's approximation

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

N_arr = np.linspace(0, 1, acc)


### ---- ANIMATION FUNCTIONS
def separatrix_shape_pos(t, a, b):      #let z = t 
    result = (a*((3/2)*(a**2)*(b**2)+(b**4)-(t**4))**(0.5))/(b**2)
    return result 

def separatrix_domain(a,b):             #domain for x-axis (where x=z=t)
    result = ((3/2)*(a**2)*(b**2) + b**4)**(1/4)
    return result 

def separatrix_range(a,b): 
    result = ((3/2)*a**2*b**2 + b**4)**(0.25)
    return result 
#######################################################################################################
#######################################################################################################
#################################                 ANIMATION               #############################
#######################################################################################################
####################################################################################################### 
    
#GRAPHING PARAMETERS 
psi_arr = []
for v in N_arr:
    psi_arr.append(internal_psi_stein_New(r_mesh, z_mesh, a, b, v))
figAnimation, axis = plt.subplots()
axis.set_xlim(-Rd, Rd)
axis.set_ylim(-Zd, Zd)
cont = plt.contourf(r_mesh, z_mesh, psi_arr[0], 0)    # first image on screen

legend_text = axis.text(
    0.05, 0.95, "", transform=axis.transAxes, 
    fontsize=10, verticalalignment="top"
)

#GRPAHING FUNCTIONS 
def update_data(frame):
    global cont 
    psi = psi_arr[frame]
    cont.remove()
    cont = axis.contourf(r_mesh, z_mesh, psi, 0)
    axis.set_title(f"Shape Index: {N_arr[frame]:.3f}")    
    return cont


animation = FuncAnimation(
                fig = figAnimation,
                func = update_data,
                frames = acc,
                interval = 5
                          )

plt.xlabel("z [m]", fontsize = labelFontSize)                 # Label for the x-axis
plt.ylabel("r [m]", fontsize = labelFontSize)     # Label for the y-axis
plt.title("Separatrix Shape Animation w/ Shape Index")
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.legend(fontsize = 'medium')
plt.grid()
animation.save("separatrix_shape.gif")
plt.show()

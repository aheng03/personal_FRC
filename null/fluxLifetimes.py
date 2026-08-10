#######################################################################################################
#######################################################################################################
#################################                  IMPORTS                 ############################
#######################################################################################################
#######################################################################################################
import matplotlib.pyplot as plt #library for plotting 
import numpy as np              #library for basic math functions 
import sys                      #library for various system-specific parameters and functions

from plottingParameters import *  
from functions4plasma import*

from scipy.optimize import fsolve
from scipy.optimize import root 
from mpl_toolkits.mplot3d import Axes3D                         #registers the 3D projection 
from matplotlib.ticker import AutoMinorLocator
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator 
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
Rw = 0.6*Lconv         # wall radius [m]
Rc = Rw                 # coil radius [m]

sig = 1.5               # flare parameter; adjustable parameter that's fixed for Steinhauer's paper
f = 1.5                 # internal psi error factor for Sporer's approximation

Xs = 0.75               # normalized separatrix ratio [\]
Rs = Xs * Rw            # separatrix radius [m]
Lx = 3.00*Lconv         # FRC length [m]
Z0 = Lx / 2             # FRC half-length [m]
B0 = 10                 # applied field B0 from Sporer's unreleased paper 
Bw = 10                 # Magnetic field at the midplane at the wall [T] from Steinhaur
B00 = Bw                # Sporer vacuum field [T] still working out how this relates to Steinhauer
a = Rs                  # FRC semi-minor axis [m]
b = Z0                  # FRC semi-major axis [m]
eps = a / b             # inverse elongation

q = ee

Nu_normal = 1.03* 10**(-4)            # classical cross-field Spitzer resistivity (N_normal_clas) from Sporer's paper about flux lifetimes 
A_brems = 1.6*10**(-38)     # from Sporer's paper about flux lifetimes [Wm^3 / sqrt(eV)]
C_tilt = 1              # ranges from 1 to 2, just set to 1 for now 

acc = 1000              #number of variables in list -- accuracy 

###----------------------------------------------------------------------------------------DOMAIN SETUP
h = 4.00*Lconv                              #length of liner [m]
domx = 1.1                                  #weight to extend the domain; helps show psi=1 curves better
Rd = Rw * domx                              #half-length of computational domain in r-dir [m]
Zd = h/2 * domx                             #half-length of computational domain in z-dir [m]

dr = 0.005*Lconv                           #mesh fidelity in r-dir [m]
dz = 0.005*Lconv                           #mesh fidelity in z-dir [m]


r = np.arange(-Rd, Rd+dr, dr)               #r-array for mesh
z = np.arange(-Zd, Zd+dz, dz)               #z-array for mesh
R, Z = np.meshgrid(r, z, indexing='ij')     #(R,Z) mesh elements

###---------------------------------------------------------------------------------FUNCTIONS

# Note: alp = alpha

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


def pressure_sporer(Be, Xs, psi_int, a, b, f):
    eps = a / b
    Pe = Be**2 / (2*mu0)
    x = 3/2 * Xs**2
    y = np.sqrt(3/2) * Xs * 8 * psi_int / (Be*a**2)
    z = f * eps**2

    P = Pe * (1 - x + y + z)
    return P

def sporer_Be(Xs, B0):
    Be = B0 / (1 - Xs**2)
    return Be


### ================================== EXTERNAL FUNCTIONS 


def external_E_system(Es, eps, Xs, sig):     #external parameters
    '''
    frc05_2D
    '''
    E0, E1, E2, alpha = Es
    eq1 = E0 - E1 * (4/eps**2) + E2 * (2*eps**2*alpha/(1-alpha**2)**2)
    eq2 = E0 + E1 + 2*alpha*E2 / np.sqrt(eps**2+alpha**2)
    e0 = eps**2*Xs**3*E2/4/sig**2
    e1 = (alpha+sig)/ np.sqrt(eps**2+Xs**2*(alpha+sig)**2)
    e2 = (alpha-sig)/ np.sqrt(eps**2+Xs**2*(alpha-sig)**2)
    e3 = 2*alpha/ np.sqrt(eps**2+(Xs*alpha)**2)
    eq3 = E1 - e0 * (e1 + e2 - e3)
    eq4 = E0 + 2*E1/Xs**2 - eps**2*alpha*Xs**3*E2 / ((alpha*Xs)**2+eps**2)*1.5 - 1
    return [eq1, eq2, eq3, eq4]


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

### ================================== INTERNAL FUNCTIONS 

def internal_psi_sporer(r, z, a, b, Be, Xs, f):     #internal flux
    eps = a / b         #Inverse elongation
    T1 = np.sqrt(3/2)
    T2 = (Xs * Be * r**2) / 2
    T3 = 1 - (r/a)**2 - (z/b)**4 + f * eps**2
    
    psi_int = T1 * T2 * T3 
    return psi_int

def internal_psi_stein(r, z, a, b, B0, B1, D0, D1):
    b0 = 1 - (r/a)**2 - (z/b)**2
    d0 = (r/a)**2 - 4*(z/a)**2
    d1 = (r/a)**4 - 12*(r*z/a/a)**2 + 8*(z/a)**4
    b1 = 1 + D0*d0 + D1*d1

    psi_int = (r**2/2) * (B0*b0 + B1*b1)
    return psi_int

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

### ================================== FLUX LIFETIME FUNCTIONS 
def tau_clas(r_s, T, lamb_A):                   #anomalous flux liftimes for calssical
    Nu = Nu_normal * lamb_A * T**(-3/2)
    result = (1/16) * (mu0*r_s**(2))/(Nu)
    return result

def tau_LSX(r_s, m, T, B0, r_w):        #anomalous flux lifetimes for LSX
    x_s = r_s/r_w
    p_ie = ((2*m*kB*T)**(0.5))/(ee*B0)  #temperature is in eV*** 
    result = (6.5 * 10**(-5)) * np.sqrt(x_s) * ((r_s)**(2.14)/(p_ie**0.5)**(2.14))
    return result

def tau_brems(n, T):                    #bremsstrahlung decay time for any hydrogenic plasma
    result = (2*n*kB*T)/(A_brems*n**(2)*np.sqrt(kB*T))*(1/np.sqrt(eV_J))
    return result

def tau_LSX_num(Rw, n, rs):                #alternative flux lifetime from LSX experiment
    xs = rs/Rw
    result = 0.02 * np.sqrt(xs) * (n**(0.53)) * (rs**(2.14))
    return result

def tau_clas_num(Rw, n, rs):                #alternative flux lifetime from LSX experiment
    xs = rs/Rw
    co = 3/(np.sqrt(xs)*n**(0.53)*rs**(0.14))
    result = (1/16) * (rs**(2))*(1/co)
    return result


### ================================== TILT FUNCTIONS 
def gamma_MHD(C_tilt, B0, elong, x_s, n, rw):    # p = mass density (use n_max, should be at the O point ( r = Rs / sqrt(2)))
    z_s = elong * x_s * rw
    V_a = B0 / (np.sqrt(mu0*n*m_dt))         # recall, B0 = external mag field 
    result = C_tilt * (V_a/z_s)
    return result

def tau_MHD(C_tilt, B0, elong, x_s, n, rw):
    fac1 = (elong * x_s * rw) / C_tilt
    fac2 = (np.sqrt(mu0 * n * m_dt)) / B0
    result = fac1 * fac2 
    return result

def gamma_tilt(gamma_MHD, T, B0, elong, xs, rw):
    v_perp = np.sqrt((kB * T)/mD) # v_perp = v_th; T and kB in eV
    p_i = (m_dt*v_perp)/(q*B0)       # ion gyroradius (in the external mag field B0)
    z_s = elong * xs * rw
    result = gamma_MHD * (np.e ** ((-3*p_i)/(z_s)))
    return result 

def tau_tilt(gamma_tilt):
    result = 1/gamma_tilt
    return result

def gamma_tilt_MHD(elong, x_s, rw, n):
    S_i = ccc * np.sqrt((m_dt * eps0) / (n * (ee**2)))
    A = (-3 * elong * S_i) / (x_s*rw)
    result = np.e**(A)
    return result


### ================================== EXTRANEOUS FUNCTIONS 
def coulomb_log(T, n):      #n is number density
    dummy1 = np.log(np.sqrt(n)*T**(-5/4))
    dummy2 = (10**(-5)*((np.log(T)-2)**2)/16)**(0.5)
    result = 23.5 - dummy1 - dummy2 
    return result

def nu_chodura(Cth, f, n, Z, e, m, j_e, T):         # all units SI
    omega_pi = np.sqrt((n*(Z**2)*(e**2))/(eps0*m))
    ve = (j_e)/(n*e)
    c_s = np.sqrt((Z*kB*T + 3*kB*T)/(m))
    multiple = (1-(np.e)**(-(ve)/(f*c_s)))
    chodura = Cth * omega_pi * multiple
    return chodura

#######################################################################################################
#######################################################################################################
#################################                 PARAMETERS               ############################
#######################################################################################################
####################################################################################################### 


###-------------------------------------------------------------------------------------EXTERNAL REGION
Eguess = [Bw, Bw/3, Bw/5, 0.9]              #initial guess for E parameters
sol = root(
    fun=external_E_params,
    x0=Eguess,
    args=(eps, Xs, sig),
    method='hybr',   # same algorithm as old fsolve, or try 'lm'
    tol=1e-6
)
if not sol.success:
    print("root() failed to converge:", sol.message)
    Br_ext = Bz_ext = 0
else:
    E0, E1, E2, alpha = sol.x
    psi_ext = external_psi(R, Z, B00, a, b, E0, E1, E2, alpha)
    dpsi__dz_ext = external_dpsi__dz(R, Z, B00, a, b, E0, E1, E2, alpha)   #check these signs
    dpsi__dr_ext = external_dpsi__dr(R, Z, B00, a, b, E0, E1, E2, alpha)
    Br_ext = -(1/R) * dpsi__dz_ext   #check these signs
    Bz_ext = (1/R) * dpsi__dr_ext


###-------------------------------------------------------------------------------------INTERNAL REGION

if model=="stein":
    psi_int = internal_psi_stein(R, Z, a, b, B0, B1, D0, D1)
    Br_int = (1/R) * internal_dpsi__dz_stein(R, Z, a, b, B0, B1, D0, D1)
    Bz_int = -(1/R) * internal_dpsi__dr_stein(R, Z, a, b, B0, B1, D0, D1)
elif model=="sporer":
    psi_int = internal_psi_sporer(R, Z, a, b, Bw, Xs, f)
    Br_int = (1/R) * internal_dpsi__dz_sporer(R, Z, a, b, Bw, Xs)
    Bz_int = -(1/R) * internal_dpsi__dr_sporer(R, Z, a, b, Bw, Xs, f)



###----------------------------------------------------------------MAGNETIC FIELD CONSTRUCTION - B(r,z)
construct = "cutoff"            #net or cutoff

if construct=="cutoff":
    #inside = (psi_int > 0)
    inside = (psi_ext < 0)
    psi = np.where(inside, psi_int, psi_ext)
    Br = np.where(inside, Br_int, Br_ext)
    Bz = np.where(inside, Bz_int, Bz_ext)
elif construct=="net":
    psi = psi_ext - psi_int
    dpsi_dr, dpsi_dz = np.gradient(psi, R, Z, edge_order=2)
    Br = -(1/R) * dpsi_dz
    Bz = (1/R) * dpsi_dr

###--------------------------------------------------------------------------------------CURRENT DENSITY
dBr__dr, dBr__dz = np.gradient(Br, r, z, edge_order=2)
dBz__dr, dBz__dz = np.gradient(Bz, r, z, edge_order=2)
J = (1 / mu0) * (dBr__dz - dBz__dr)     #current density [A/?^2]
Jphi = J

###--------------------------------------------------------------------------------------PRESSURE
if model=="sporer":
    P_int = pressure_sporer(Bw, Xs, psi_int, a, b, f)       #his Be is my (and stein's) Bw
    P = np.where(inside, P_int, 0)
elif model=="stein":
    #P = pressure_stein(r, z, Jphi, Br, Bz)
    P_int = pressure_jeff_mesh(R, Z, -Jphi, Br, Bz)
    P = np.where(inside, P_int, 0)
    
# Pressure Mesh Unaveraged
Pscaled_unavg = (P)/(Bw**2/(2*mu0))

# Pressure Mesh Average
P_avg_inside = np.mean(P[inside])       # Shape: (numTemps,)
P_mesh = np.full_like(P, P_avg_inside)  # Creating a copy of P, except every value is P_avg_inside
Pscaled_avg = (P_mesh)/(Bw**2/(2*mu0))             #equation 22 from improved analytical equilibrium paper, for pressure

###--------------------------------------------------------------------------------------NUMBER DENSITY
Nr, Nz = R.shape

#temperature values
temperature_list = np.linspace(100, 500, num=5)  #eV

numTemps = len(temperature_list)
n_array_unavg = np.zeros(((Nr, Nz, numTemps)))
n_array_avg = np.zeros(((Nr, Nz, numTemps)))

n_array_max_unavg = np.zeros(numTemps)
n_array_max_avg = np.zeros(numTemps)


# Unaveraged Pressure Mesh 
for i in range(numTemps):
    T = temperature_list[i] * eV/kB
    n_array_unavg[:,:,i] = P / (kB * T)
    n = P / (kB * T)
    n_array_max_unavg[i] = np.max(n_array_unavg[:,:,i]) #finding the max for each temperature slice
    #n_max_array[i] = np.max(n_array) --> finding the max number density over entire array

n_array_max_tau_Punavg = n_array_max_unavg* (10**(-21))    # Converting to the units of the LSX number density flux equation, specific to only said equation
n_array_max_log_Punavg = n_array_max_unavg * 10 **(-6)     # Units from m^-3 to cm^-3
   
# Averaged Pressure Mesh
for i in range(numTemps):
    T = temperature_list[i] * eV/kB
    n_array_avg[:,:,i] = P_mesh / (kB * T)
    n = P_mesh / (kB * T)
    n_array_max_avg[i] = np.max(n_array_avg[:,:,i])     # Finding the max for each temperature slice
    #n_max_array[i] = np.max(n_array) --> finding the max number density over entire array

n_array_max_tau_Pavg = n_array_max_avg* (10**(-21))    # Converting to the units of the LSX number density flux equation, specific to only said equation
n_array_max_log_Pavg = n_array_max_avg * 10 **(-6)     # Units from m^-3 to cm^-3 ; used for Coulomb's logarithm


# output test zone 
print(P_mesh)
print(P)
print("pressure averaged number density array:", n_array_max_avg)
print("pressure unaveraged number density array:", n_array_max_unavg)
print("n_array_max_tau_Pavg: ", n_array_max_tau_Pavg)
print("n_array_max_tau_Puanvg: ", n_array_max_tau_Punavg)


#Coulomb values for corresponding T values
coulomb_list = []
for index, temp in enumerate(temperature_list): 
    coulomb_list.append(coulomb_log(temp, n_array_max_log_Pavg[index]))
    
###--------------------------------------------------------------------------------------GRAPHING PARAMETERS    


#elongation (b/a) data points between 1 and 10
elongation_list = np.linspace(1,10,num= acc)

#inverse elongation (a/b)
eps_list = 1/elongation_list
b_list = np.linspace(0.0001, 0.0534, num = acc)
a_list = b_list*eps_list                        #a == rs 
Xs_list = a_list / Rw
Xs_listValues = np.linspace(0.3,0.9,num=5)      #X_s values between 0.3 and 0.9 for Shape Index
    

#y_lists for tau_clas for every temperature value 
ylist_tau_clas = []
for index, value in enumerate(temperature_list): 
    ylist_tau_clas.append(tau_clas(a_list, value, coulomb_list[index]))


#y_lists for tau_LSX for every temperature value 
ylist_tau_LSX = []
for i in temperature_list: 
    empty_list = []
    for index, value in enumerate(elongation_list):
        empty_list.append(tau_LSX(a_list[index], mD, i, B0, Rw))
    ylist_tau_LSX.append(empty_list)


#y_list for tau_num & tau_brems 

ylist_tau_LSX_num_avg = []   # Pressure mesh avg 
ylist_tau_clas_num_avg = []

ylist_tau_LSX_num_unavg = []  # Pressure mesh unaveraged 
ylist_tau_clas_num_unavg = []
ylist_tau_brems = []            # Bremsstrahlung radiation ylist 

for index, temp in enumerate(temperature_list): 
    ylist_tau_clas_num_unavg.append(tau_clas_num(Rw, n_array_max_tau_Punavg[index], a_list))
    ylist_tau_clas_num_avg.append(tau_clas_num(Rw, n_array_max_tau_Pavg[index], a_list))
    ylist_tau_LSX_num_avg.append(tau_LSX_num(Rw, n_array_max_tau_Pavg[index], a_list))
    ylist_tau_LSX_num_unavg.append(tau_LSX_num(Rw, n_array_max_tau_Punavg[index], a_list))
    ylist_tau_brems.append(tau_brems(n_array_max_unavg[index], temp))
    
    
###--------------------------------------------------------------------------------------GRAPHING PARAMETERS (10/25) 
# tilt lifetimes
ylist_gamma_MHD = []
ylist_gamma_tilt = [] 
ylist_tau_tilt = []
ylist_tau_MHD = []

for index, value in enumerate(n_array_max_unavg): 
    dummy1 = [] 
    dummy2 = [] 
    dummy3 = []
    dummy4 = []
    for e_index, e_value in enumerate(elongation_list):
        gamma_MHD_value = gamma_MHD(C_tilt, B0, e_value, Xs, value, Rw)  
        dummy1.append(gamma_MHD_value)
        tau_MHD_value = tau_MHD(C_tilt, B0, e_value, Xs, value, Rw) 
        dummy4.append(tau_MHD_value)
        gamma_tilt_value = gamma_tilt(gamma_MHD_value, temperature_list[index], B0, e_value, Xs, Rw)
        dummy2.append(gamma_tilt_value)
        dummy3.append(tau_tilt(gamma_tilt_value))
    ylist_tau_tilt.append(dummy3)
    ylist_gamma_tilt.append(dummy2)
    ylist_gamma_MHD.append(dummy1)
    ylist_tau_MHD.append(dummy4)


### --- normalized list init 
ylist_tau_clas_norm = []
ylist_tau_LSX_norm = []
ylist_brem_norm = []
ylist_gamma_tilt_MHD = []

# taking it just for each temperature value 
for idx in range(len(temperature_list)):
    dummy1 = [] 
    dummy2 = [] 
    dummy3 = []
    dummy4 = []
    for value in range(len(elongation_list)):
        dummy1.append(ylist_tau_clas[idx][value] / ylist_tau_MHD[idx][value])
        dummy2.append(ylist_tau_LSX[idx][value] / ylist_tau_MHD[idx][value])
        dummy3.append(ylist_tau_brems[idx] / (10**(2) * ylist_tau_MHD[idx][value]))
        dummy4.append(gamma_tilt_MHD(elongation_list[value], Xs, Rw, n_array_max_unavg[idx]))
    ylist_tau_LSX_norm.append(dummy1)
    ylist_tau_clas_norm.append(dummy2)
    ylist_brem_norm.append(dummy3)
    ylist_gamma_tilt_MHD.append(dummy4)


#######################################################################################################
#######################################################################################################
#################################                 PLOTS                    ############################
#######################################################################################################
####################################################################################################### 

plot_num = 0            #indexing for plot #

'''
### --- elongation (b/a) is the x-axis, tau_clas is the y-axis
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))

    plt.figure(num = plot_num, dpi = dpi_res)
    plt.plot(elongation_list, ylist_tau_clas[T_index])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{classical}$ vs Elongation, T = ' + temp + "eV", fontsize = titleFontSize)
    plt.xlabel("Elongation [b/a]", fontsize = labelFontSize)            # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    title_save = "tau_clas_T" + temp + " vs elongation"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 

### --- Shape Index (N3) is the x-axis, tau_class is the y-axis
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)

    for index, value in enumerate(Shape_Index_N3_list):
        lab_el = "{:.2f}".format(Xs_listValues[index])
        plt.plot(value, ylist_tau_clas[T_index], label = lab_el )

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{classical}$ vs Shape Index, T = ' + temp + "eV", fontsize = titleFontSize)
    plt.xlabel("Shape Index [N3]", fontsize = labelFontSize)  # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    plt.legend(title = r'$X_s$', fontsize = 'small')
    title_save = "tau_clas_T" + temp + " vs Shape Index"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 

#######################################################################################################
####################################################################################################### 


### --- elongation (b/a) is the x-axis, tau_LSX is the y-axis
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.plot(elongation_list, ylist_tau_LSX[T_index])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{LSX}$ vs Elongation, T = ' + temp + "eV", fontsize = titleFontSize)
    plt.xlabel("Elongation (b/a)", fontsize = labelFontSize)  # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    title_save = "tau_LSX_T" + temp + " vs Elongation"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 

### --- Shape Index (N3) is the x-axis, tau_LSX is the y-axis

for T_index, T_value in enumerate(temperature_list): 
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi= dpi_res)

    for index, value in enumerate(Shape_Index_N3_list):
        lab_el = "{:.2f}".format(Xs_listValues[index])
        plt.plot(value, ylist_tau_LSX[T_index], label = lab_el )  
        
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{LSX}$ vs Shape Index, T = ' + temp + "eV", fontsize = titleFontSize)
    plt.xlabel("Shape Index [N3]", fontsize = labelFontSize)  # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    plt.legend(title = r'$X_s$', fontsize = 'small')
    title_save = "tau_LSX_T" + temp + " vs Shape Index"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1

'''

#######################################################################################################
####################################################################################################### 
#################################           COMPARISON PLOTS               ############################
#######################################################################################################
#######################################################################################################

colors = {
    "clas": "C0",        # first color in the cycle
    "LSX": "C1",         # second color
    "unavg": "C2",        # third color
    "avg": "C3",         # fourth color
}


### --- Normalized Confinement Lifetime Graph 
# --- Finding the peaks 
def find_x_peak(x_list, y_list, y_list_peak):
    for index, value in enumerate(x_list): 
        if y_list[index] == y_list_peak:
            return value
peaks_clas = max(ylist_tau_clas_norm[0])
peak_clas_elong = find_x_peak(elongation_list, ylist_tau_clas_norm[0], peaks_clas)
valley_clas = min(ylist_tau_clas_norm[0])
valley__clas_elong = find_x_peak(elongation_list, ylist_tau_clas_norm[0], valley_clas)

peaks_LSX = max(ylist_tau_LSX_norm[0])
peak_LSX_elong = find_x_peak(elongation_list, ylist_tau_LSX_norm[0], peaks_LSX)
valley_LSX = min(ylist_tau_LSX_norm[0])
valley__LSX_elong = find_x_peak(elongation_list, ylist_tau_LSX_norm[0], valley_LSX)



fig, ax1 = plt.subplots(num=plot_num, dpi=dpi_res, figsize=(9, 6))      # width=9in, height=6in
ax1.set_xlim(1, 10)
ax1.set_ylim(0,20)
ax1.grid()

# Primary axis (tau ratios)
ax1.plot(elongation_list, ylist_tau_clas_norm[0], color=colors['clas'], label=r'$\tau_{clas} \ / \ \tau_{MHD}$', linewidth=lineWidth, )
ax1.plot(elongation_list, ylist_tau_LSX_norm[0], color=colors['LSX'], label=r'$\tau_{LSX} \ / \ \tau_{MHD}$', linewidth=lineWidth)
ax1.plot(elongation_list, ylist_brem_norm[0], color = colors['unavg'], label=r'$\tau_{Brem} \ / \ 100\tau_{MHD}$', linewidth=lineWidth)

ax1.ticklabel_format(axis='y', style='plain', scilimits=(0,0))
x_ticks = np.arange(1, int(elongation_list.max()) + 1, 1)
ax1.set_xticks(x_ticks)
ax1.set_yticks(np.arange(0,20+1,2))
ax1.spines['bottom'].set_position(('data', 0))
ax1.set_xticklabels(ax1.get_xticks(), weight = 'bold')
ax1.tick_params(axis='x', which='major')
ax1.set_yticklabels(ax1.get_yticks(), weight = 'bold')
ax1.set_xlabel("Elongation", fontsize=labelFontSize, weight='bold')
ax1.set_ylabel(r'$\tau_{\Phi} \ / \ \tau_{MHD}$', fontsize=labelFontSize, weight='bold')


# Secondary axis (gamma_tilt / gamma_MHD)
ax2 = ax1.twinx()
ax2.plot(elongation_list, ylist_gamma_tilt_MHD[0], color='red', label=r'$\gamma_{tilt} \ / \ \gamma_{MHD}$', linewidth=lineWidth, linestyle='dashed')

y2_ticks = np.arange(0, 1.1, 0.1)
ax2.set_yticks(y2_ticks)
ax2.set_ylabel(r'$\gamma_{tilt}  \ / \ \gamma_{MHD}$', color='red', fontsize=labelFontSize, labelpad=10, weight='bold')
ax2.tick_params(axis='y', labelcolor='red')
ax2.ticklabel_format(axis='y', style='plain', scilimits=(0,0))
ax2.set_yticklabels(ax2.get_yticks(), weight = 'bold')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Adjust spacing to avoid label overlap
fig.subplots_adjust(right=0.82)  # moves the right side of the plot inward a bit


# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

# Place legend outside the plot area
ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
           fontsize = legendFontSize,
           loc='upper right',
           frameon=True,
           facecolor='white',
           edgecolor='black',
           framealpha=1)
title_save = f"normalized confinement lifetimes vs elongation, T = {int(temperature_list[0])} eV"
plt.savefig(title_save + ".png", dpi=dpi_res, bbox_inches='tight')
plot_num += 1


print(f"maximuim classical confinement value: ({peak_clas_elong}, {peaks_clas})" )
print(f"maximum LSX confinement value:({peak_LSX_elong}, {peaks_LSX})" )
print(f"minimum classical confinement value: ({valley__clas_elong}, {valley_clas})" )
print(f"minimum LSX confinement value:({valley__LSX_elong}, {valley_LSX})" )
print("Inputs: \n" \
"B0:", B0, "T \n", \
"T:", temperature_list[0], "eV \n", \
"n0:", n_array_max_unavg[0], "m^-3 \n",\
"Xs:", Xs, "\n",\
"Rw:", Rw, "m \n", \
"E: an array from 1 to 10 with 100 data points")

'''
### --- 10/25/25 Normalized Flux Lifetime Plot against Elongation for T value = 100 eV
plt.figure(num = plot_num, dpi = dpi_res)
plt.grid()
plt.plot(elongation_list, ylist_tau_clas_norm[0], color = colors['clas'], label = r'$\frac{\tau^{\Phi}_{clas}}{\tau_{MHD}}$') 
plt.plot(elongation_list, ylist_tau_LSX_norm[0], color = colors['LSX'], label = r'$\frac{\tau^{\Phi}_{LSX}}{\tau_{MHD}}$')
plt.plot(elongation_list, ylist_brem_norm[0], color = 'red', label = r'$\frac{\tau^{\Phi}_{brem}}{\tau_{MHD}}$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.title(r'$\tau^{\Phi}_{clas\_norm}$, $\tau^{\Phi}_{LSX\_norm}$, $\tau^{\Phi}_{brems\_norm}$ vs Elongation', fontsize = titleFontSize)
plt.xlabel("Elongation [b/a]", fontsize = labelFontSize)                # Label for the x-axis
plt.ylabel(r'$\frac{\tau^{\Phi}}{\tau_{MHD}}$', fontsize = labelFontSize)     # Label for the y-axis
plt.legend(fontsize = 'small')
title_save = "normalized confinement lifetimes vs elongation, T = 100 eV"
plt.savefig(title_save + ".png", dpi = dpi_res)
plot_num += 1


plt.figure(num = plot_num, dpi = dpi_res)
plt.grid()
plt.plot(elongation_list, ylist_gamma_tilt_MHD[0], color = 'red', label = r'$\frac{\gamma_{tilt}}{\gamma_{MHD}}$')
plt.title(r'$\frac{\gamma_{tilt}}{\gamma_{MHD}}$ vs Elongation', fontsize = titleFontSize)
plt.xlabel("Elongation [b/a]", fontsize = labelFontSize)                # Label for the x-axis
plt.ylabel(r'$\frac{\tau^{\Phi}}{\tau_{tilt}}$', fontsize = labelFontSize)     # Label for the y-axis
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend(fontsize = 'small')
title_save = "normalized gamma_tilt vs elongation"
plt.savefig(title_save + ".png", dpi = dpi_res)
'''
'''

### --- tau_clas & tau_LSX vs Elongation 
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.grid()

    plt.plot(elongation_list, ylist_tau_clas[T_index], color = colors['clas'], label = r'$\tau^{\Phi}_{clas}$') 
    plt.plot(elongation_list, ylist_tau_LSX[T_index], color = colors['LSX'], label = r'$\tau^{\Phi}_{LSX}$')
    plt.plot(elongation_list, ylist_tau_LSX_num_avg[T_index], color = colors['avg'], label = r'$\tau^{\Phi}_{Punavg}$')
    plt.plot(elongation_list, ylist_tau_LSX_num_unavg[T_index], color = colors['unavg'], label = r'$\tau^{\Phi}_{Pavg}$')

    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    plt.title(r'$\tau^{\Phi}_{clas}$ & $\tau^{\Phi}_{LSX}$ vs Elongation, T = ' + temp + " eV")
    plt.xlabel("elongation [b/a]", fontsize = labelFontSize)
    plt.ylabel(r'Flux lifetime [s], $\tau$', fontsize = labelFontSize)
    plt.legend(fontsize = 'small')
    title_save = "tau_clas, tau_LSX vs elongation, T = " + temp + "eV"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 

### --- tau_clas vs tau_clas number density vs Elongation
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.grid()

    plt.plot(elongation_list, ylist_tau_clas[T_index], color = colors['clas'], label = r'$\tau^{\Phi}_{clas}$') 
    plt.plot(elongation_list, ylist_tau_clas_num_avg[T_index], color = colors['avg'], label = r'$\tau^{\Phi}_{clas-avg}$')
    plt.plot(elongation_list, ylist_tau_clas_num_unavg[T_index], color = colors['unavg'], label = r'$\tau^{\Phi}_{clas-unavg}$')

    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    plt.title(r'$\tau^{\Phi}_{clas}$, $\tau^{\Phi}_{num-avg}$, $\tau^{\Phi}_{unavg}$ vs Elongation, T = ' + temp + " eV")
    plt.xlabel("elongation [b/a]", fontsize = labelFontSize)
    plt.ylabel(r'Flux lifetime [s], $\tau$', fontsize = labelFontSize)
    plt.legend(fontsize = 'small')
    title_save = "tau_clas, tau_clas_num vs elongation, T = " + temp + "eV"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 

### --- tau_LSX vs tau_LSX number density vs Elongation
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.grid()

    plt.plot(elongation_list, ylist_tau_LSX[T_index], color = colors['LSX'], label = r'$\tau^{\Phi}_{LSX}$') 
    plt.plot(elongation_list, ylist_tau_LSX_num_avg[T_index], color = colors['avg'], label = r'$\tau^{\Phi}_{avg}$')
    plt.plot(elongation_list, ylist_tau_LSX_num_unavg[T_index], color = colors['unavg'], label = r'$\tau^{\Phi}_{unavg}$')

    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    plt.title(r'$\tau^{\Phi}_{LSX}$, $\tau^{\Phi}_{avg}$, $\tau^{\Phi}_{unavg}$ vs Elongation, T = ' + temp + " eV")
    plt.xlabel("elongation [b/a]", fontsize = labelFontSize)
    plt.ylabel(r'Flux lifetime [s], $\tau$', fontsize = labelFontSize)
    plt.legend(fontsize = 'small')
    title_save = "tau_LSX, tau_LSX_num vs elongation, T = " + temp + "eV"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 
    
#######################################################################################################
####################################################################################################### 

### --- tau_clas & tau_LSX vs Shape Index
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.grid()

    for index, value in enumerate(Shape_Index_N3_list): 
        plt.plot(value, ylist_tau_clas[T_index], 
                color=colors["clas"], 
                label = r'$\tau^{\Phi}_{clas}$' if index ==0 else "")
            
    for index, value in enumerate(Shape_Index_N3_list): 
        plt.plot(value, ylist_tau_LSX[T_index], 
                color=colors["LSX"], 
                label = r'$\tau^{\Phi}_{LSX}$' if index ==0 else "")
    
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{clas}$ & $\tau^{\Phi}_{LSX}$ vs Shape Index, T = ' + temp + " eV")
    plt.xlabel("Shape Index [N3]", fontsize = labelFontSize)                 # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    plt.legend(fontsize = 'small')
    title_save = "tau_clas, tau_LSX vs Shape Index, T = " + temp + "eV"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 
   
    
### --- tau_LSX & tau_num density (LSX) vs Shape Index
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.grid()
            
    for index, value in enumerate(Shape_Index_N3_list): 
        plt.plot(value, ylist_tau_LSX[T_index], 
                color=colors["LSX"], 
                label = r'$\tau^{\Phi}_{LSX}$' if index ==0 else "")
          
    for index, value in enumerate(Shape_Index_N3_list):
        plt.plot(value, ylist_tau_LSX_num_avg[T_index], 
                color=colors["avg"], 
                label = r'$\tau^{\Phi}_{num-avg}$' if index ==0 else "")
        
    for index, value in enumerate(Shape_Index_N3_list):
        plt.plot(value, ylist_tau_LSX_num_unavg[T_index], 
                color = colors["max"], 
                label = r'$\tau^{\Phi}_{num-max}$' if index ==0 else "")     
        
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{LSX}$ & $\tau^{\Phi}_{num}$ vs Shape Index, T = ' + temp + " eV")
    plt.xlabel("Shape Index [N3]", fontsize = labelFontSize)                 # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    plt.legend(fontsize = 'small')
    title_save = "tau_LSX, tau_num vs Shape Index, T = " + temp + "eV"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1    
    
### --- tau_clas & tau_num density (clas) vs Shape Index
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.grid()
            
    for index, value in enumerate(Shape_Index_N3_list): 
        plt.plot(value, ylist_tau_clas[T_index], 
                color=colors["clas"], 
                label = r'$\tau^{\Phi}_{clas}$' if index ==0 else "")
          
    for index, value in enumerate(Shape_Index_N3_list):
        plt.plot(value, ylist_tau_clas_num_avg[T_index], 
                color=colors["avg"], 
                label = r'$\tau^{\Phi}_{num-avg}$' if index ==0 else "")
        
    for index, value in enumerate(Shape_Index_N3_list):
        plt.plot(value, ylist_tau_clas_num_avg[T_index], 
                color = colors["unavg"], 
                label = r'$\tau^{\Phi}_{num-max}$' if index ==0 else "")     
        
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{clas}$ & $\tau^{\Phi}_{num}$ vs Shape Index, T = ' + temp + " eV")
    plt.xlabel("Shape Index [N3]", fontsize = labelFontSize)                 # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    plt.legend(fontsize = 'small')
    title_save = "tau_clas, tau_num vs Shape Index, T = " + temp + "eV"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 
'''

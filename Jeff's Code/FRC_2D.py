#FRC 2D Equilibrium 

#################################################
#####               IMPORTS                 #####
#################################################
import numpy as np
import matplotlib.pyplot as plt
import cmath
import sys
import time
from plottingParameters import *
from scipy.optimize import fsolve
from scipy.optimize import root
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from matplotlib.ticker import AutoMinorLocator
from matplotlib.path import Path
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LogNorm
from scipy.integrate import cumulative_trapezoid as cumtrapz

#################################################
#####               CONSTANTS               #####
#################################################
mu0 = 1.26 * 10**(-6)
eV = 1.6022*10**(-19)
kb = 1000




t0code = time.perf_counter()
#################################################
#####               FUNCTIONS               #####
#################################################
###----------------------------------------------------------------------------------EXTERNAL FUNCTIONS
def external_psi(r, z, Bw, a, b, E0, E1, E2, alpha):
    psi_w = Bw * a**2 / 2
    e0 = (r/a)**2
    e1 = (r/a)**2 * ((r/a)**2 - 4*(z/a)**2)
    denom1 = (alpha*b+z)**2 + r**2
    denom2 = (alpha*b-z)**2 + r**2
    e2 = (alpha*b+z)/np.sqrt(denom1) + (alpha*b-z)/np.sqrt(denom2)
    
    psi = psi_w * (E0*e0 + E1*e1 + E2*e2)
    return psi



def external_dpsi__dr(r, z, Bw, a, b, E0, E1, E2, alpha):
    denom1 = (alpha*b-z)**2 + r**2
    denom2 = (alpha*b+z)**2 + r**2
    T0 = E0 * r
    T1 = E1 * (r/a**2) * (2*r**2 - 4*z**2)
    T2 = E2 * (a**2/2) * (r*(alpha*b-z)/(denom1**1.5) + r*(alpha*b+z)/(denom2**1.5))

    dpsi__dr = Bw * (T0 + T1 - T2)
    return dpsi__dr



def external_dpsi__dz(r, z, Bw, a, b, E0, E1, E2, alpha):
    denom1 = (alpha*b-z)**2 + r**2
    denom2 = (alpha*b+z)**2 + r**2
    T1 = E1 * 8*z*r**2/a**4
    T2 = E2 * ((z-alpha*b)**2/(denom1**1.5) - 1/denom1**0.5 + 1/denom2**0.5 - (alpha*b+z)**2/denom2**1.5)
    
    dpsi__dz = Bw * a**2/2 * (T2 - T1)
    return dpsi__dz



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



def external_Be(Bw, a, b, Xs):
    Be__Bw = 1 + 1.46 * (3.2 + ((b/a) / (1.5-Xs))**4)**(-1)
    Be = Bw * Be__Bw
    return Be



def sporer_Be(Xs, B0):
    Be = B0 / (1 - Xs**2)
    return Be




###----------------------------------------------------------------------------------INTERNAL FUNCTIONS
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



def internal_psi_sporer(r, z, a, b, Be, Xs, f):
    eps = a / b         #Inverse elongation
    T1 = np.sqrt(3/2)
    T2 = (Xs * Be * r**2) / 2
    T3 = 1 - (r/a)**2 - (z/b)**4 + f * eps**2
    
    psi_int = T1 * T2 * T3 
    return psi_int



def internal_dpsi__dr_sporer(r, z, a, b, Be, Xs, f):
    eps = a / b
    F = np.sqrt(3/2) * Xs * Be * r
    P = 1 - 2*(r/a)**2 - (z/b)**4 + f * eps**2
    
    dpsi__dr = F * P
    return dpsi__dr
    


def internal_dpsi__dz_sporer(r, z, a, b, Be, Xs):
    dpsi__dz = - np.sqrt(6) * Xs * Be * r**2 * z**3 / b**4
    return dpsi__dz



def internal_B0(a, b, Xs, Bw):
    N = 2.56 - 3.28*Xs
    D = 9.65 - 15.2*Xs + (b/a)**4
    p = 1 + N/(Xs**2 * D)    
    sr = np.sqrt(3/2)*Xs

    B0 = Bw * sr * p
    return B0



def shape_index_N(a, b, Rs):
    N = b**2 / a / Rs
    return N



def shape_index_N2(eps, E0, E1, E2, alpha):
    lamb = (1 + (eps/alpha)**2)**(-0.5)
    N1 = 4 * eps * E1
    N2 = 3 * alpha * lamb**5 * E2
    D1 = -eps * E1
    D2 = alpha * lamb * E2 * (2 + lamb**2)

    N = (N1 + N2) / (eps**2 * (D1 + D2))
    return N



def shape_index_N3(a, b, Xs):
    N = (1 + (0.8+Xs**2) * (b/a-1)**2)**(-1)
    return N



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




###-------------------------------------------------------------------------------------OTHER FUNCTIONS
def pressure_sporer(Be, Xs, psi_int, a, b, f):
    eps = a / b
    Pe = Be**2 / (2*mu0)
    x = 3/2 * Xs**2
    y = np.sqrt(3/2) * Xs * 8 * psi_int / (Be*a**2)
    z = f * eps**2

    P = Pe * (1 - x + y + z)
    return P



def pressure_jeff_mesh(R, Z, Jphi, Br, Bz):
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

def pressure_stein(r, z, Jphi, Br, Bz):
    Nr, Nz = Jphi.shape
    
    dP__dr = Jphi * Bz          #partial of pressure wrt r [Pa/m]
    dP__dz = - Jphi * Bz        #partial of pressure wrt z [Pa/m]

    P = np.zeros((Nr, Nz))      #initialize pressure array [Pa]

    r_rev = r[::-1]             #reverse r‐axis arrays so that "index 0" in the reversed array is r_max
    dr_rev = np.diff(r_rev)     #differences along reversed array

    for j in range(Nz):
        col_dPdr = dP__dr[:, j]          # Extract column j of dP/dr and reverse it:
        col_dPdr_rev = col_dPdr[::-1]

        # Cumulatively integrate col_dPdr_rev from r_max downward:
        #    cumtrapz(y, x, initial=0) returns an array Y of length len(x),
        #    where Y[k] = ∫_{x[0]}^{x[k]} y(x') dx'.
        #
        # Here x = r_rev, y = col_dPdr_rev, and x[0] = r_rev[0] = r_max.
        # So cumtrapz(col_dPdr_rev, r_rev, initial=0) gives:
        #   integrated[k] = ∫_{r_max}^{ r_rev[k] } dP/dr * d(r').
        # Because r_rev is decreasing, this is effectively ∫_{r_rev[k]}^{r_max}.
        #
        integral_rev = cumtrapz(col_dPdr_rev, r_rev, initial=0.0)
        # By construction, integral_rev[0] = 0 at r_rev[0] = r_max.

        # Now flip back to the original r‐ordering:
        #   P[:, j] = - integral from r_i to r_max = -integral_rev[ reversed index ]
        P[:, j] = - integral_rev[::-1]

    return P




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


# ===========================
# Example usage:

# Suppose you have psi, R, Z already defined on your mesh:
#    psi = ...   # 2D array shape (Nr,Nz)
#    R, Z = np.meshgrid(r, z, indexing='ij')

# To get only the longest loop at psi = 1.0:
#r0, z0 = get_flux_contours(psi, R, Z, psi_level_mag=1.0, return_all=False)

# To get all disjoint loops at psi = 1.0:
#all_loops = get_flux_contours(psi, R, Z, psi_level_mag=1.0, return_all=True)
# all_loops is now something like [(r1,z1), (r2,z2), …]

# If you just want to plot them:
#for (r_i, z_i) in all_loops:
#    plt.plot(r_i, z_i, '-k', linewidth=1)
#
#plt.gca().set_aspect('equal')
#plt.show()


#######################################################################################################
#######################################################################################################
#################################                  MAIN                    ############################
#######################################################################################################
#######################################################################################################
Lconv = 1e-2
###------------------------------------------------------------------------------------------PARAMETERS
Rw = 0.61*Lconv        #wall radius [m]
Rc = Rw                 #coil radius [m]
Xs = 0.75               #normalized separatrix ratio [\]
Rs = Xs * Rw            #separatrix radius [m]

h = 4.00*Lconv         #length of liner [m]
Lx = 3.00*Lconv        #FRC length [m]
Z0 = Lx / 2             #FRC half-length [m]
a = Rs                  #FRC semi-minor axis [m]
b = Z0                  #FRC semi-major axis [m]

sig = 1.5               #flare parameter; adjustable parameter that's fixed for Steinhauer's paper
f = 1.5                 #internal psi error factor for Sporer's approximation
eps = a / b             #inverse elongation

Bw = 10                 #Magnetic field at the midplane at the wall [T] from Steinhaur
B00 = Bw                #Sporer vacuum field [T] still working out how this relates to Steinhauer



###----------------------------------------------------------------------------------------DOMAIN SETUP
domx = 1.1                                  #weight to extend the domain; helps show psi=1 curves better
Rd = Rw * domx                              #half-length of computational domain in r-dir [m]
Zd = h/2 * domx                             #half-length of computational domain in z-dir [m]

dr = 0.005*Lconv                           #mesh fidelity in r-dir [m]
dz = 0.005*Lconv                           #mesh fidelity in z-dir [m]

r = np.arange(-Rd, Rd+dr, dr)               #r-array for mesh
z = np.arange(-Zd, Zd+dz, dz)               #z-array for mesh
R, Z = np.meshgrid(r, z, indexing='ij')     #(R,Z) mesh elements


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

N = shape_index_N3(a, b, Xs)
D0 = internal_D0(eps)
D1 = internal_D1(eps)
B0 = internal_B0(a, b, Xs, Bw)
B1 = internal_B1(B0, D0, D1, N, eps)
Be = external_Be(Bw, a, b, Xs)
Bee = sporer_Be(Xs, B00)    #Sporer Be [T]



###-------------------------------------------------------------------------------------INTERNAL REGION
model = "stein"

if model=="stein":
    psi_int = internal_psi_stein(R, Z, a, b, B0, B1, D0, D1)
    Br_int = (1/R) * internal_dpsi__dz_stein(R, Z, a, b, B0, B1, D0, D1)
    Bz_int = -(1/R) * internal_dpsi__dr_stein(R, Z, a, b, B0, B1, D0, D1)
elif model=="sporer":
    psi_int = internal_psi_sporer(R, Z, a, b, Bw, Xs, f)
    Br_int = (1/R) * internal_dpsi__dz_sporer(R, Z, a, b, Bw, Xs)
    Bz_int = -(1/R) * internal_dpsi__dr_sporer(R, Z, a, b, Bw, Xs, f)


mid_z = np.argmin(np.abs(z))
mid_r = int(len(r)/2)


###----------------------------------------------------------------------------------------------OUTPUT
print("FRC PARAMETERS---------------------------------------------------------------------------------")
print("\t Rw = {:.4f} [cm] (Conducting Wall Radius)".format(Rw/Lconv))
print("\t Rs = {:.4f} [cm] (Spearatrix Radius)".format(Rs/Lconv))
print("\t Xs = {:.4f} (Normalized Separatrix Radius)".format(Xs))
print(" ")

print("\t E = {:.4f} (FRC Elongation)".format(1/eps))
print("\t eps = {:.4f} (Inverse Elongation)".format(eps))
print("\t a = {:.4f} = [cm] (FRC Radius)".format(a/Lconv))
print("\t b = {:.4f} [cm] (FRC Half Length)".format(b/Lconv))
print("\t sig = {:.2f} (Adjustable Parameter)".format(sig))
print(" ")


print("\t N = {:.4f} (Shape Index; 0=racetrack  1=ellipse)".format(N))
print("\t E0 = {:.4f} ".format(E0))
print("\t E1 = {:.4f} ".format(E1))
print("\t E2 = {:.4f} ".format(E2))
print("\t alpha = {:.4f} (0 < alpha < 1)".format(alpha))
print(" ")

print("\t B0 = {:.3f} [T]".format(B0))
print("\t B1 = {:.3f} [T]".format(B1))
print("\t Bw = {:.3f} [T]".format(Bw))
print("\t Be = {:.3f} [T]".format(Be))
print("\t D0 = {:.4f} ".format(D0))
print("\t D1 = {:.4f} ".format(D1))
print(" ")



# — build a single string with all your FRC parameters —
summary_text = (
    "FRC PARAMETERS\n"
    f"Rc = {Rw/Lconv:.4f} [cm] (Conducting Wall Radius)\n"
    f"Rs = {Rs/Lconv:.4f} [cm] (Separatrix Radius)\n"
    f"Xs = {Xs:.4f} (Normalized Separatrix Radius)\n\n"
    f"E  = {1/eps:.4f} (FRC Elongation)\n"
    f"eps = {eps:.4f} (Inverse Elongation)\n"
    f"a  = {a/Lconv:.4f} [cm] (FRC Radius)\n"
    f"b  = {b/Lconv:.4f} [cm] (FRC Half Length)\n"
    f"sig = {sig:.2f} (Adjustable Parameter)\n\n"
    f"N  = {N:.4f} (Shape Index; 0=racetrack 1=ellipse)\n"
    f"E0 = {E0:.4f}\n"
    f"E1 = {E1:.4f}\n"
    f"E2 = {E2:.4f}\n"
    f"alpha = {alpha:.4f} (0 < alpha < 1)\n\n"
    f"B0 = {B0:.3f} [T]\n"
    f"B1 = {B1:.3f} [T]\n"
    f"Bw = {Bw:.3f} [T]\n"
    f"Be = {Be:.3f} [T]\n"
    f"D0 = {D0:.4f}\n"
    f"D1 = {D1:.4f}"
)



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

print("BIGGGGGGG:", psi_ext)
# find every negative ψ
neg_vals = psi[psi < 0]

# print them
#print("All negative psi values:")
#print(neg_vals)

rSep, zSep = get_flux_contours(psi, R, Z, 0)
r0_comb, z0_comb = get_flux_contours(psi, R, Z, 0)
r1_comb, z1_comb = get_flux_contours(psi_int, R, Z, 1)

zidx = np.where(np.isclose(z,Z0))[0][0]
#print(zidx, psi[0,zidx])



###------------------------------------------------------------------------------------------------DIV(B)
dBr__dr, dBr__dz = np.gradient(Br, r, z, edge_order=2)
dBz__dr, dBz__dz = np.gradient(Bz, r, z, edge_order=2)

idx = 30
idy = idx
divB = dBr__dr + Br/R + dBz__dz
divB_scaled = np.abs(divB) / (Bw/Rs)
print("Divergence of B:")
print("\tdBr__dr \t Br/R \t\t dBz__dz \t divB \t\t divB_scaled")
print(
    f"\t{dBr__dr[idx,idy]} "
    f"{Br[idx,idy]/R[idx,idy]} "
    f"{dBz__dz[idx,idy]} "
    f"{divB[idx,idy]} "
    f"{divB_scaled[idx,idy]}"
)
print("\t divB_scaled max and min", divB_scaled.max(), divB_scaled.min())





###--------------------------------------------------------------------------------------CURRENT DENSITY
J = (1 / mu0) * (dBr__dz - dBz__dr)     #current density [A/?^2]
Jphi = J





###---------------------------------------------------------------------------------------------PRESSURE
dP__dr = Jphi * Bz          #partial of pressure wrt radius [Pa/m]
dP__dz = - Jphi * Br        #partial of pressure wrt z [Pa/m]

if model=="sporer":
    P_int = pressure_sporer(Bw, Xs, psi_int, a, b, f)       #his Be is my (and stein's) Bw
    P = np.where(inside, P_int, 0)
elif model=="stein":
    #P = pressure_stein(r, z, Jphi, Br, Bz)
    P_int = pressure_jeff_mesh(R, Z, -Jphi, Br, Bz)
    P = np.where(inside, P_int, 0)

Pscaled = P/(Bw**2/(2*mu0))



###------PLOTTING
fig, ax = plt.subplots(figsize=(5,8))
#levels = np.linspace(np.nanmin(Pscaled), np.nanmax(Pscaled), 50)
levels = np.linspace(np.nanmin(Pscaled), 1.5, 50)

#ax.plot(rSep, zSep, 'w-', linewidth=1, zorder=2)
cf = ax.contourf(R/Lconv, Z/Lconv, Pscaled, levels=levels, cmap='plasma')
cbar = fig.colorbar(cf, ax=ax)
#cbar.set_label("$P(r,z)$  $[Pa]$", fontsize=12)
cbar.set_label("$P(r,z)/\\frac{B_w^2}{2\\mu_0}$", fontsize=12)

ax.set_xlabel("$r$  $[cm]$")
ax.set_ylabel("$z$  $[cm]$")
title_clean = "Pressure"
ax.set_title(title_clean)
ax.set_aspect('equal')
plt.tight_layout()
#plt.show()


####Error Check
# Compute numerical partials of your reconstructed P(r,z):
dPdr_check, dPdz_check = np.gradient(P, r, z, edge_order=2)

# Compare to the original dP__dr, dP__dz:
print("Pressure:")
print("\tMax Pressure = {:.3e} [Pa]".format(P_int.argmax()))
print("\t Max |dPdr_check - dP__dr| =", np.nanmax(np.abs(dPdr_check - dP__dr)))
print("\t Max |dPdz_check - dP__dz| =", np.nanmax(np.abs(dPdz_check - dP__dz)))


###--------------------------------------------------------------------------------------NUMBER DENSITY
Nr, Nz = R.shape

delta_T = 100       #[eV]
T_low = delta_T
T_high = 500
T_high = T_high + delta_T
T_array = np.arange(T_low, T_high, delta_T)

numTemps = len(T_array)
n_array = np.zeros(((Nr, Nz, numTemps)))
n_array_all_scaled = np.zeros(((Nr, Nz, numTemps)))
n_array_each_scaled = np.zeros(((Nr, Nz, numTemps)))
n_max_array = np.zeros(numTemps)

for i in range(numTemps):
    T = T_array[i] * eV/kb
    n_array[:,:,i] = P / (kb * T)
    n = P / (kb * T)
    n_max_array[i] = np.max(n_array)

n_min = np.min(n_array)
n_max = np.max(n_array)
levels = np.linspace(n_min, n_max, 100)
levels_scaled = np.linspace(n_min/n_max, n_max/n_max, 100)

print("Number Density:")
print("\t n_max = {:.3e} [m^-3] = {:.3e} [cm^-3]".format(n_max, n_max*Lconv**3))
print("\t n_min = {:.3e} [m^-3] = {:.3e} [cm^-3]".format(n_min, n_min*Lconv**3))


###---plotting
fig, axes = plt.subplots(
    1, numTemps,
    figsize=(14, 8),
    sharex=True, sharey=True,
    constrained_layout=True
)
for i, ax in enumerate(axes):
    n = n_array[:,:,i]    

    cf = ax.contourf(
        R/Lconv,           # convert m→cm if you like
        Z/Lconv,
        n,    # the 2D density at T_array[i]
        levels=levels,
        cmap='plasma'
    )
    # r‐axis label on every plot:
    ax.set_xlabel(r"$r\;[\mathrm{cm}]$")
    # only the first panel gets a z‐axis label; all others drop their left ticks
    if i == 0:
        ax.set_ylabel(r"$z\;[\mathrm{cm}]$")
    else:
        ax.tick_params(labelleft=False)

    # optional title per panel showing T:
    ax.set_title(f"$T = {T_array[i]:.0f}$ $eV$", fontsize=10)
    ax.set_aspect('equal')

cbar = fig.colorbar(
    cf,
    ax=axes,                # pass the list of all subplots
    orientation='vertical',
    fraction=0.02,          # width of cbar relative to figure
    pad=0.02
)
cbar.set_label(r"$n(r,z)$ $[m^{-3}]$", fontsize=12)
title_clean = "Number Density"
title_save = str(re.sub(r'[\\/*?:"<>|\n$]', '_', title_clean))

fig.savefig(title_save + ".png", dpi=dpi_res)
#fig.savefig(title_save + ".pdf", dpi=dpi_res)
#plt.show()



###----------------------------------------------------Individual Plots
for i in range(numTemps):
    n = n_array[:,:,i]
    n_min = np.min(n)
    n_max = np.max(n)
    levels = np.linspace(n_min, n_max, 100)
    levels_scaled = np.linspace(n_min/n_max, n_max/n_max, 100)
    
    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 8),
        sharex=True, sharey=True,
        constrained_layout=True
    )

    cf0 = axes[0].contourf(
        R/Lconv,           # convert m→cm if you like
        Z/Lconv,
        n,    # the 2D density at T_array[i]
        levels=levels,
        cmap='plasma'
    )
    cf1 = axes[1].contourf(
        R/Lconv,           # convert m→cm if you like
        Z/Lconv,
        n/np.max(n_array[:,:,i]),    # the 2D density at T_array[i]
        levels=levels_scaled,
        cmap='plasma'
    )

    cbar0 = fig.colorbar(
        cf0,
        ax=axes[0],                # pass the list of all subplots
        orientation='vertical',
        pad=0.05
    )
    cbar1 = fig.colorbar(
        cf1,
        ax=axes[1],                # pass the list of all subplots
        orientation='vertical',
        pad=0.05
    )

    for j in range(2):
        if j == 0:
            axes[j].set_ylabel(r"$z\;[\mathrm{cm}]$")
        else:
            axes[j].tick_params(labelleft=False)
        axes[j].set_aspect('equal')
        axes[j].set_xlabel(r"$r\;[\mathrm{cm}]$")

    cbar0.set_label(r"$n(r,z)$ $[m^{-3}]$", fontsize=12)
    cbar1.set_label(r"$n(r,z)/n_{peak}$ $[m^{-3}]$", fontsize=12)
    title_clean = f"Number Density at T = {T_array[i]:.0f} eV"
    title_save = str(re.sub(r'[\\/*?:"<>|\n$]', '_', title_clean))
    fig.suptitle(title_clean, fontsize=16)
    plt.savefig(title_save + ".png", dpi=dpi_res)
    plt.close(fig)



###--------------------------------------------------------------------------------------TEMPERATURE


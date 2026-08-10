import sys 
import numpy as np
from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]  # goes up from FRC to code
sys.path.append(str(base_dir))

from _functions4plasma import *

### === FLUX LIFETIME FUNCTIONS 
def tau_clas(r_s, T, lamb_A):                   #anomalous flux liftimes for calssical
    Nu = Nu_normal * lamb_A * T**(-3/2)
    return ((1/16) * (mu0*r_s**(2))/(Nu))


def tau_LSX(r_s, m, T, B0, x_s):        #anomalous flux lifetimes for LSX
    p_ie = ((2*m*kB*T)**(0.5))/(ee*B0)  #temperature is in eV*** 
    return ((6.5 * 10**(-5)) * np.sqrt(x_s) * ((r_s)**(2.14)/(p_ie**0.5)**(2.14)))


def tau_brems(n, T):                    #bremsstrahlung decay time for any hydrogenic plasma
    return (2*np.sqrt(kB*T)*np.sqrt(eV_J))/(A_brems*n)


def tau_LSX_num(Rw, n, rs):                #alternative flux lifetime from LSX experiment
    xs = rs/Rw
    result = 0.02 * np.sqrt(xs) * (n**(0.53)) * (rs**(2.14))
    return result


def tau_clas_num(Rw, n, rs):                #alternative flux lifetime from LSX experiment
    xs = rs/Rw
    co = 3/(np.sqrt(xs)*n**(0.53)*rs**(0.14))
    result = (1/16) * (rs**(2))*(1/co)
    return result


### === TILT FUNCTIONS 
def gamma_tilt(C_tilt, B0, elong, x_s, n, rw):    # p = mass density (use n_max, should be at the O point ( r = Rs / sqrt(2)))
    z_s = elong * x_s * rw
    V_a = B0 / (np.sqrt(mu0*n*m_dt))         # recall, B0 = external mag field 
    return C_tilt * (V_a/z_s)


def gamma_rot(T, Bw, k):
    fac1 = (k**2)/(8*np.pi**2)
    fac2 = T / Bw
    return fac1*fac2

def gamma_tear(tau_clas, n, B0, T, Xs, Rw, eps):
    '''
    n = number density 
    lamb = wavelength 
    '''

    R_max = Xs*Rw       # equal to Rs -- separatrix radius 
    R_min = (-R_max*(eps-1))/(eps+1)
    R0 = (R_max + R_min) / 2


    rho_io = np.sqrt(2*m_dt*kB*T)/(ee*B0)
    S = R0/rho_io

    nu_A = B0 / np.sqrt(mu0*n)
    delt_i = ccc*np.sqrt((m_dt*eps0) / (n*ee**2))

    lamb = 2*np.pi*delt_i*(S**(1/4))
    k = 2*np.pi / lamb
    
    return (tau_clas**(-3/5)) * (nu_A/k)**(2/5) * delt_i**(-4/5)


def tau_tilt(gammaTilt):
    return 1/gammaTilt


def tau_rot(gammaRot):
    return 1/gammaRot


def tau_tear(gammaTear):
    return 1/gammaTear


### ================================== EXTRANEOUS FUNCTIONS 
def coulomb_log(T, n):      #n is number density
    dummy1 = np.log(np.sqrt(n)*T**(-5/4))
    dummy2 = (10**(-5)*((np.log(T)-2)**2)/16)**(0.5)
    return 23.5 - dummy1 - dummy2


def nu_chodura(Cth, f, n, Z, j_e, T):         # all units SI
    omega_pi = np.sqrt((n*(Z**2)*(ee**2))/(eps0*m_dt))
    ve = (j_e)/(n*ee)
    c_s = np.sqrt((Z*kB*T + 3*kB*T)/(m_dt))
    multiple = (1-(np.e)**(-(ve)/(f*c_s)))
    chodura = Cth * omega_pi * multiple
    return chodura
import numpy as np

#######################################################################################################
#######################################################################################################
#################################          Universal Constants             ############################
#######################################################################################################
#######################################################################################################
# UNIT CONVERSION
eV          = 1.6022*10**(-19)          # amount of Joules in a eV
eV_K        = 11604.525                 # amount of kelvin / eV
eV_J        = 1/(6.2415*10**(18))           # amount of Joule / eV
amu         = 1.66e-27                  # atomic mass unit to kg


# CONSTANTS
ccc         = 2.99*10**(8)              # speed of light [m/s]
eps0        = 8.85*10**(-12)            # permittivity constant [C^2 / N*m^2]
mu0         = np.pi * 4e-7
kb          = 1.3806*10**(-23)          # Boltzmann constant (in SI) 
kB          = kb * eV_K                 # Boltzmann constant [J/eV]
ee          = 1.6 *10**(-19)            # Fundamental Charge [C]

m_e         = 5.486e-4*amu              #mass of electron [kg]
mH          = 1.007*amu                 # mass of proton [kg]
mD          = 2.014*amu                 # mass of deuterium [kg]
mT          = 3.016*amu                 # mass of tritium [kg]
m_dt        = (mD + mT) / 2             # 50/50 mass of D-T


#######################################################################################################
#######################################################################################################
#################################          Specific Constants              ############################
#######################################################################################################
#######################################################################################################
Nu          = 10**(-4)           #plasma parameter (N_normal_clas) from Sporer's paper about flux lifetimes 

sig         = 1.5
f           = 1.5

Nu_normal   = 1.03* 10**(-4)            # classical cross-field Spitzer resistivity (N_normal_clas) from Sporer's paper about flux lifetimes 
A_brems     = 1.6*10**(-38)     # from Sporer's paper about flux lifetimes [Wm^3 / sqrt(eV)]
C_tilt      = 1              # ranges from 1 to 2, just set to 1 for now 

import numpy as np

domdim = "2D"
Lconv = 1e3
acc = 2
deg = 20


Rw                      = 6.1/Lconv        # Wall Radius [m]
Rc = Rw                 # coil radius [m]

zLen                    = 150/Lconv              #liner length [m]
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
Np          = 1                         #number of points in phi-direction [\]
dp          = (Pmax - Pmin) / Np        #azimuthal differential [rad]

Zmax        = zLen / 2                  #maximum z-domain value [m]
Zmin        = -zLen / 2                 #minimum z-domain value [m]
Nz          = 1000                      #number of points in z-direction [\]
dz          = (Zmax - Zmin) / Nz        #axial differential [m]

domx        = 1.0                       #domain multiplier to make plotting modifications easier
domy        = 0.8                       #domain multiplier to make plotting modifications easier
Rd          = Rmax * domx               #domain radius [m]
Zd          = Zmax * domy               #domain z-length [m]


if domdim=='1D':
    r = np.arange(dr, Rd+dr, dr)

elif domdim=='2D':
    r = np.arange(-Rd, Rd+dr, dr)
    z = np.arange(-Zd, Zd+dz, dz)    
    R, Z = np.meshgrid(r, z, indexing='ij')
    Rcc = 0.5 * (R[:-1, :-1] + R[1:, 1:])
    Zcc = 0.5 * (Z[:-1, :-1] + Z[1:, 1:])

elif domdim=='3D':
    r = np.arange(-Rd, Rd+dr, dr)
    p = np.arange(Pmin, Pmax+dp, dp)
    z = np.arange(-Zd, Zd+dz, dz)
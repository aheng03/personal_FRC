# Date of Last Update: 6.23.2026
# Author: Adelina Hengyucius
# Grad Shafranov Implementation
'''
Solves general 2D plasma equilibria based on methodology from "'One Size Fits All'"
which uses an analytical solution to the Grad Shafranov Equation.

Requires
--------
True/False Input: 
- Up-Down Symmtery 
- Smoothness
- Double Null Divertor 

Parameter Input:
- A
- Triangularity
- Elongation
- Normalized Separatrix Ratio 
'''

### === LIBRARIES
import time 
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]  # goes up from FRC to code
sys.path.append(str(base_dir))

### === FILES 
from gs_func import*
from _plottingParameters import *
from _linerParameters import *

start_time = time.time()

class _constants():
    '''
    Defining and solving for unknown constants by declaring class for given conditions
    '''
    def __init__(self, pivots, bc_parameters_d, bc_word_d, num, updownsym, Smooth, a, e, k, A):
        '''
        pivots:             # of independent columns
        bc_parameters_d:    dict in format of (BC #: COORD VALUE)
        bc_word_d:          dict in format of (BC #: NAME OF BC)
        num:                # of BCs
        updownsym:          T/F based off user input
        smooth:             T/F based off user input
        a:                  alpha as found from delta (user input)
        e:                  epsilon (user input)
        k:                  kappa (found from user input)
        A:                  A (user input)

        The below refers to the derivative of the function taken with respect to
        0 - NONE
        1 - X
        2 - XX
        3 - Y
        4 - YY
        '''
        self.pivots = pivots
        self.bc_parameters_d = bc_parameters_d
        self.bc_word_d = bc_word_d
        self.num = num              # num_BC
        self.updownsym = updownsym
        self.Smooth = Smooth
        self.a = a  # alpha 
        self.e = e  # epsilon
        self.k = k  # kappa
        self.A = A  # A
        self.N_coeff = find_N(a, e, k)
        self.bc_dict = {}
        self.con_dict = {}
        self.homo_dict = {}          # homogeneous solution dictionary - phi_h
        self.bc_homo_dict = {}       # homogeneous solution (bc) dictionary - phi_h, bc
        self.parti_dict = {}         # particular solution dictionary - phi_p
        self.bc_parti_dict = {}      # particular solution (bc) dictionary - phi_p, bc
        self.diff_parti_dict = {}    # particular solution difference - (phi_p - phi_p, bc)
        
        self.homo_matrix = []        # homgeneous matrix - list of lists
        self.bc_homo_matrix = []     # homogenous (bc) matrix - list of lists
        self.diff_parti_vec = []     # particular solution difference - vector form 
        self.constant_vec = []       # constant vector
        self.word_to_ord = {
            "OUTER_EQUATORIAL_POINT" : 0,
            "INNER_EQUATORIAL_POINT" : 0,
            "HIGH_POINT" : 0, 
            "LOWER_X_POINT" : 0,
            "OUTER_EQUATORIAL_POINT_UP_DOWN_SYM" : 3,
            "INNER_EQUATORIAL_POINT_UP_DOWN_SYM" : 3,
            "HIGH_POINT_MAXIMUM" : 1,
            "B_N=0" : 1, 
            "B_T=0" : 3,
            "OUTER_EQUATORIAL_POINT_CURVATURE" : 4,
            "INNER_EQUATORIAL_POINT_CURVATURE" : 4, 
            "HIGH_POINT_CURVATURE" : 2
        }
        # Objects
        self.var_phi = poly_homo(updownsym)
        self.var_phi_x = poly_homo_x(updownsym)
        self.var_phi_xx = poly_homo_xx(updownsym)
        self.var_phi_y = poly_homo_y(updownsym)
        self.var_phi_yy = poly_homo_yy(updownsym)


    def dictionary_values(self):
        '''
        Appending all values to the dictionaries (representative of matrices) accordingly
        '''
        for key in self.bc_parameters_d:
            word = self.bc_word_d[key]
            x = self.bc_parameters_d[key][0]
            y = self.bc_parameters_d[key][1]
            self.homo_dict.setdefault(key, [])
            self.bc_homo_dict.setdefault(key, [])
            var_parti = parti_solu(word)
            ord = self.word_to_ord[self.bc_word_d[key]]
            self.parti_dict[key] = var_parti.boundary_condition(x, y, self.A)
            self.bc_parti_dict[key] = var_parti.bc_boundary_condition(x, y, self.A)
            
            if ord == 0:
                obj_h = self.var_phi
            elif ord == 1:
                obj_h = self.var_phi_x
            elif ord == 2:
                obj_h = self.var_phi_xx
                obj_bc = self.var_phi_y
            elif ord == 3:
                obj_h = self.var_phi_y
            elif ord == 4:
                obj_h = self.var_phi_yy
                obj_bc = self.var_phi_x
            for poly_func in self.num:
                self.homo_dict[key].append(obj_h.poly_psi(poly_func, x, y))
                if ord in (0, 1, 3):
                    self.bc_homo_dict[key].append(0)
                else:
                    self.bc_homo_dict[key].append(obj_bc.poly_psi(poly_func, x, y))
    
    def N_constants(self):
        '''
        Multiplying the N constants into the bc_homogeneous dictionary and bc_parti dictionary
        prior to transferring it into a matrix.
        '''
        for index in self.bc_homo_dict:
            word = self.bc_word_d[index]
            if word == "OUTER_EQUATORIAL_POINT_CURVATURE":
                self.bc_homo_dict[index] = [value * (-self.N_coeff[0]) for value in self.bc_homo_dict[index]]
                self.bc_parti_dict[index] = (-self.N_coeff[0] * self.bc_parti_dict[index])
            elif word == "INNER_EQUATORIAL_POINT_CURVATURE":
                self.bc_homo_dict[index] = [value * (-self.N_coeff[1]) for value in self.bc_homo_dict[index]]
                self.bc_parti_dict[index] = (-self.N_coeff[1] * self.bc_parti_dict[index])
            elif word == "HIGH_POINT_CURVATURE":
                self.bc_homo_dict[index] = [value * (-self.N_coeff[2]) for value in self.bc_homo_dict[index]]
                self.bc_parti_dict[index] = (-self.N_coeff[2] * self.bc_parti_dict[index])

    def constants(self):
        '''
        Solving for constants
        '''
        # Transferring values from dictionaries into matrices & solving for the difference
        # in particular values
        for key in self.parti_dict:
            self.diff_parti_dict[key] = self.parti_dict[key] - self.bc_parti_dict[key]
        for key in self.bc_homo_dict:
            self.bc_homo_matrix.append(self.bc_homo_dict[key])
            self.homo_matrix.append(self.homo_dict[key])
        
        # phi_bc - phi_h 
        matrix_diff = [[self.bc_homo_matrix[i][j] - self.homo_matrix[i][j]
                   for j in range(len(self.homo_matrix[0]))] for i in range(len(self.homo_matrix))]
        # con_columns are the independent columns index values (independent poly func)
        inverse_matrix, con_columns = find_inverse_matrix(
            matrix_diff, self.homo_dict, self.bc_homo_dict, self.diff_parti_dict, self.bc_word_d, self.pivots)
        # Transferring the particular vector from dictionary to list
        for value in self.diff_parti_dict.values():
            self.diff_parti_vec.append(value)
        # solving for the constant vector accordingly 
        for lst in inverse_matrix:
            con_sum = 0
            for index in range(len(lst)): 
                con_sum += lst[index] * self.diff_parti_vec[index]
            self.constant_vec.append(con_sum)
        for key in self.homo_dict:
            sum = self.parti_dict[key]
            for index, value in enumerate(self.homo_dict[key]):
                con = self.constant_vec[index]
                sum += (con * value)
            self.bc_dict[key] = sum
        for index, value in enumerate(con_columns):
            self.con_dict[value] = self.constant_vec[index]
    
    def ret_values(self):
        self.dictionary_values()
        self.N_constants()
        self.constants()
        return self.con_dict, self.bc_dict


def calculate_flux(Axis, Smooth, Double, A, eps, kap, delta, R, Z):
    '''
    Smooth: T/F (user input
    Axis:   T/F (user input)
    Double: T/F (user input) 
    A:      float (user input)
    eps:    float (user input)
    kap:    float (from user input)
    delta:  float (user input)
    R:      np.array (2D); (from user input)
    Z:      np.array (2D); (from user input)


    Returns 2D mesh of magnetic flux for R, Z mesh as calculated by GS
    '''
    all_piv = {i for i in range(12)}  # column pivots
    num_BC = range(0, 12)
    x_sep = 1 - 1.1*delta*eps
    y_sep = -1.1*kap*eps
    alp = alp_delt(delta)
    if Axis and Smooth:
        I_bc_word_dict = {
        0: "OUTER_EQUATORIAL_POINT", 
        1: "INNER_EQUATORIAL_POINT", 
        2: "HIGH_POINT",
        3: "OUTER_EQUATORIAL_POINT_UP_DOWN_SYM", 
        4: "INNER_EQUATORIAL_POINT_UP_DOWN_SYM", 
        5: "HIGH_POINT_MAXIMUM",
        6: "OUTER_EQUATORIAL_POINT_CURVATURE",
        7: "INNER_EQUATORIAL_POINT_CURVATURE",
        8: "HIGH_POINT_CURVATURE"
                        }
        I_boundary_cond_para_dict = {
        0: (1+eps, 0), 
        1: (1-eps, 0), 
        2: (1-delta*eps, kap*eps),
        3: (1+eps, 0), 
        4: (1-eps, 0), 
        5:(1-delta*eps, kap*eps),
        6: (1+eps, 0), 
        7: (1-eps, 0),
        8:(1-delta*eps, kap*eps)
                        }
    elif Axis and not Smooth:
        I_bc_word_dict = {
        0: "OUTER_EQUATORIAL_POINT", 
        1: "INNER_EQUATORIAL_POINT", 
        2: "HIGH_POINT",
        3: "OUTER_EQUATORIAL_POINT_UP_DOWN_SYM", 
        4: "INNER_EQUATORIAL_POINT_UP_DOWN_SYM", 
        5: "HIGH_POINT_MAXIMUM",
        6: "B_T=0",
        7: "OUTER_EQUATORIAL_POINT_CURVATURE",
        8: "INNER_EQUATORIAL_POINT_CURVATURE",
        9: "HIGH_POINT_CURVATURE"
                        }
        I_boundary_cond_para_dict = {
        0: (1+eps, 0), 
        1: (1-eps, 0), 
        2: (1-delta*eps, kap*eps),
        3: (1+eps, 0), 
        4: (1-eps, 0), 
        5:(1-delta*eps, kap*eps),
        6: (1-delta*eps, kap*eps),
        7: (1+eps, 0), 
        8: (1-eps, 0),
        9:(1-delta*eps, kap*eps)
                        }
    elif Double:
        I_bc_word_dict = {
        0: "OUTER_EQUATORIAL_POINT", 
        1: "INNER_EQUATORIAL_POINT", 
        2: "HIGH_POINT", 
        3: "B_N=0", 
        4: "B_T=0", 
        5: "OUTER_EQUATORIAL_POINT_CURVATURE",
        6: "INNER_EQUATORIAL_POINT_CURVATURE", 
        }
        I_boundary_cond_para_dict = {
        0: (1+eps, 0), 
        1: (1-eps, 0), 
        2: (x_sep, y_sep),
        3: (x_sep, y_sep), 
        4: (x_sep, y_sep), 
        5:(1+eps, 0),
        6: (1-eps, 0), 
            }
    elif not axisSymmetric:
        I_bc_word_dict = {
        0: "OUTER_EQUATORIAL_POINT", 
        1: "INNER_EQUATORIAL_POINT", 
        2: "HIGH_POINT", 
        3: "LOWER_X_POINT",
        4: "OUTER_EQUATORIAL_POINT_UP_DOWN_SYM", 
        5: "INNER_EQUATORIAL_POINT_UP_DOWN_SYM", 
        6: "HIGH_POINT_MAXIMUM",
        7: "B_N=0", 
        8: "B_T=0", 
        9: "OUTER_EQUATORIAL_POINT_CURVATURE",
        10: "INNER_EQUATORIAL_POINT_CURVATURE", 
        11: "HIGH_POINT_CURVATURE"
        }
        I_boundary_cond_para_dict = {
        0: (1+eps, 0), 
        1: (1-eps, 0), 
        2: (1-delta*eps, kap*eps),
        3: (x_sep, y_sep), 
        4: (1+eps, 0), 
        5: (1-eps, 0), 
        6:(1-delta*eps, kap*eps),
        7: (x_sep, y_sep), 
        8: (x_sep, y_sep), 
        9: (1+eps, 0), 
        10: (1-eps, 0),
        11:(1-delta*eps, kap*eps)}

    constant = _constants(all_piv, I_boundary_cond_para_dict, I_bc_word_dict, num_BC, Axis, Smooth, alp, eps, kap, A)
    I_con_dict, I_bc_dict = constant.ret_values()
    return flux(R, Z, A, I_con_dict, I_bc_word_dict, Axis)


if __name__ == '__main__':
    '''
    If this file is ran directly, then UI will pop up.
    otherwise: 
        won't run if imported to another file
    '''
    from gs_ui import *

    ### === Calculated from User Input:
    R_max = Xs*Rw       # equal to Rs -- separatrix radius 
    R_min = (-R_max*(eps-1))/(eps+1)
    R0 = (R_max + R_min) / 2 

    a = (R_max - R_min) / 2
    kap = E*2

    ### === DOMAIN SETUP
    domx = 1.1                                 # weight to extend the domain; helps show psi=1 curves better
    Rd = Rw
    Zd = kap*a * domx
    dr = 0.005/Lconv                           # mesh fidelity in r-dir [m]
    dz = 0.005/Lconv                           # mesh fidelity in z-dir [m]
    r = np.arange(-Rd, Rd+dr, dr)               #r-array for mesh
    z = np.arange(-Zd, Zd+dz, dz)               #z-array for mesh   
    r = np.delete(r, np.where(r==0))
    z = np.delete(z, np.where(z==0))
    x_ar = r/R0
    y_ar = z/R0
    R, Z = np.meshgrid(x_ar, y_ar, indexing='ij')     #(R,Z) mesh elements
    Nr, Nz = R.shape


    flux_val = calculate_flux(axisSymmetric, smooth, double_null, A, eps, kap, delt, R, Z)
    flux_min = flux_val.min()
    

    ### === Plotting 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')  # equal scaling on both axes
    dl = -(flux_min / 7)
    numColors = 11
    levels = np.linspace(flux_min, 0, numColors)
    colors = plt.cm.rainbow(np.linspace(0, 1, numColors))
    handles = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(levels))]
    labels  = [fr'$\psi={lev:.1e}$' for lev in levels]
    
    cs = ax.contour(
        R, Z, flux_val,
        levels=levels,
        cmap='rainbow',
        linewidths=1.5)
    ax.grid(alpha = 0.5)
    ax.legend(handles, labels,
            loc='center left', bbox_to_anchor=(1.1, 0.5),
            frameon=False, borderaxespad=0.0)
    ax.set_xlabel(r'$\boldsymbol{r}/\boldsymbol{R}_{\boldsymbol{0}}$')
    ax.set_ylabel(r'$\boldsymbol{z}/\boldsymbol{R}_{\boldsymbol{0}}$')
    ax.set_ylim((-Zd*Lconv / 1.5, Zd*Lconv / 1.5))
    ax.set_title(r'$\boldsymbol{Flux}, \ \boldsymbol{\psi(x,y)}$')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    fig.savefig(f"flux", dpi=dpi_res)


    print("--- %s seconds ---" % (time.time() - start_time))

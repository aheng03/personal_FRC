# Date of Last Update:  3.11.26
# Author: Adelina Hengyucius
# Grad Shafranov Functions & Classes

### === LIBRARIES
import numpy as np
import sympy as sp
tol= 1e-8

def safe_log(x, tol=tol):
    '''
    Determines whether or not ln x will be undefined
    '''
    x = np.asarray(x)
    out = np.ones_like(x)
    mask = np.abs(x) >= tol
    out[mask] = np.log(np.abs(x[mask]))
    return out

### === Homogeneous Polynomial Solution
class poly_homo:
    def __init__(self, updownsym):
        self.updownsym = updownsym

    def poly_psi_0(self, x, y):
        return 1

    def poly_psi_1(self, x, y):
        return x**2

    def poly_psi_2(self, x, y):
        logx = safe_log(x)
        return y**2 - (x**2)*logx

    def poly_psi_3(self, x, y):
        return x**4 - 4*(x**2)*(y**2)

    def poly_psi_4(self, x, y):
        logx = safe_log(x)
        return 2*(y**4) - 9*(y**2)*(x**2) + 3*(x**4)*logx - 12*(x**2)*(y**2)*logx

    def poly_psi_5(self, x, y):
        return (x**6) - 12*(x**4)*(y**2) + 8*(x**2)*(y**4)

    def poly_psi_6(self, x, y):
        logx = safe_log(x)
        term1 = 8*(y**6) - 140*(y**4)*(x**2) + 75*(y**2)*(x**4) - 15*(x**6)*logx
        term2 = 180*(x**4)*(y**2)*logx - 120*(x**2)*(y**4)*logx
        return term1 + term2

    def poly_psi_7(self, x,y):
        return y

    def poly_psi_8(self, x, y):
        return (x**2)*y

    def poly_psi_9(self, x, y):
        logx = safe_log(x)
        return y**3 - 3*(x**2)*y*logx

    def poly_psi_10(self, x, y):
        return 3*(x**4)*y - 4*(x**2)*(y**3)

    def poly_psi_11(self, x, y):
        logx = safe_log(x)
        return 8*(y**5) - 45*(x**4)*y - 80*(x**2)*(y**3)*logx + 60*(x**4)*y*logx
    
    def poly_psi(obj, index, x, y):
        if obj.updownsym and index in (7, 8, 9, 10, 11):
            return 0
        elif index == 0:
            return obj.poly_psi_0(x,y)
        elif index == 1:
            return obj.poly_psi_1(x,y)
        elif index == 2:
            return obj.poly_psi_2(x,y)
        elif index == 3:
            return obj.poly_psi_3(x,y)
        elif index == 4:
            return obj.poly_psi_4(x,y)
        elif index == 5:
            return obj.poly_psi_5(x,y)
        elif index == 6:
            return obj.poly_psi_6(x,y)
        else:
            if index == 7:
                return obj.poly_psi_7(x,y)
            elif index == 8:
                return obj.poly_psi_8(x,y)
            elif index == 9:
                return obj.poly_psi_9(x,y)
            elif index == 10:
                return obj.poly_psi_10(x,y)
            elif index == 11:
                return obj.poly_psi_11(x,y)

class poly_homo_x:
    '''
    1st order partial derivative of the polynomial homogenous 
    functions with respect to x
    '''
    def __init__(self, updownsym):
        self.updownsym = updownsym

    def poly_psi_0_x(self, x, y):
        return 0

    def poly_psi_1_x(self, x, y):
        return 2*x

    def poly_psi_2_x(self, x, y):
        x = abs(x)
        if x < tol: 
            return -2*x - x
        return -2*x*np.log(x) - x

    def poly_psi_3_x(self, x, y):
        return 4*(x**3) - 8*x*(y**2)

    def poly_psi_4_x(self, x, y):
        x = abs(x)
        if x < tol: 
            return -30*x*(y**2) + 12*(x**3) + 3*(x**3) - 24*x*(y**2)
        return -30*x*(y**2) + 12*(x**3)*np.log(x) + 3*(x**3) - 24*x*(y**2)*np.log(x)

    def poly_psi_5_x(self, x, y):
        return 6*(x**5) - 48*(x**3)*(y**2) + 16*x*(y**4)

    def poly_psi_6_x(self, x, y):
        x = abs(x)
        if x < tol: 
            part1 = -400*x*(y**4) + 480*(x**3)*(y**2) - 90*(x**5) - 15*(x**5)
            part2 = 720*(x**3)*(y**2) - 240*x*(y**4)
        else:
            part1 = -400*x*y**4 + 480*x**3*y**2 - 90*x**5*np.log(x) - 15*x**5
            part2 = 720*x**3*y**2*np.log(x) - 240*x*y**4*np.log(x)
        return part1 + part2

    def poly_psi_7_x(self, x, y):
        return 0

    def poly_psi_8_x(self, x, y):
        return 2*x*y

    def poly_psi_9_x(self, x, y):
        x = abs(x)
        if x < tol: 
            return -6*x*y - 3*x*y
        return -6*x*y*np.log(x) - 3*x*y

    def poly_psi_10_x(self, x, y):
        return 12*(x**3)*y - 8*x*(y**3)

    def poly_psi_11_x(self, x, y):
        x = abs(x)
        if x < tol: 
            return -120*(x**3)*y - 160*x*(y**3) - 80*x*(y**3) + 240*(x**3)*y
        return -120*x**3*y - 160*x*y**3*np.log(x) - 80*x*y**3 + 240*x**3*y*np.log(x)
    
    def poly_psi(obj, index, x, y):
        if obj.updownsym and index in (7, 8, 9, 10, 11):
            return 0
        elif index == 0:
            return obj.poly_psi_0_x(x,y)
        elif index == 1:
            return obj.poly_psi_1_x(x,y)
        elif index == 2:
            return obj.poly_psi_2_x(x,y)
        elif index == 3:
            return obj.poly_psi_3_x(x,y)
        elif index == 4:
            return obj.poly_psi_4_x(x,y)
        elif index == 5:
            return obj.poly_psi_5_x(x,y)
        elif index == 6:
            return obj.poly_psi_6_x(x,y)
        else:    
            if index == 7:
                return obj.poly_psi_7_x(x,y)
            elif index == 8:
                return obj.poly_psi_8_x(x,y)
            elif index == 9:
                return obj.poly_psi_9_x(x,y)
            elif index == 10:
                return obj.poly_psi_10_x(x,y)
            elif index == 11:
                return obj.poly_psi_11_x(x,y)
            
class poly_homo_xx: 
    '''
    2nd order partial derivative of the polynomial homogenous 
    functions with respect to x
    '''
    def __init__(self, updownsym):
        self.updownsym = updownsym

    def poly_psi_0_xx(self, x, y):
        return 0

    def poly_psi_1_xx(self, x, y):
        return 2

    def poly_psi_2_xx(self, x, y):
        if abs(x) < tol: 
            return -5
        x = abs(x)
        return -2*np.log(x) - 3

    def poly_psi_3_xx(self, x, y):
        return 12*(x**2) - 8*(y**2)

    def poly_psi_4_xx(self, x, y):
        x = abs(x)
        if x < tol: 
            return -30*(y**2) + 36*(x**2) + 21*(x**2) - 24*(y**2)
        return -30*(y**2) + 36*(x**2)*np.log(x) + 21*(x**2) - 24*(y**2)*np.log(x)

    def poly_psi_5_xx(self, x, y):
        return 30*(x**4) - 144*(x**2)*(y**2) + 16*(y**4)

    def poly_psi_6_xx(self, x, y):
        x = abs(x)
        if x < tol: 
            part1 = -640*(y**4) + 2160*(x**2)*(y**2) - 450*(x**4)
            part2 = -165*(x**4) + 2160*(x**2)*(y**2) - 240*(y**4)
            return part1 + part2
        part1 = -640*y**4 + 2160*x**2*y**2 - 450*x**4*np.log(x)
        part2 = -165*x**4 + 2160*x**2*y**2*np.log(x) - 240*y**4*np.log(x)
        return part1+part2

    def poly_psi_7_xx(self, x, y):
        return 0

    def poly_psi_8_xx(self, x, y):
        return 2*y

    def poly_psi_9_xx(self, x, y):
        x = abs(x)
        if x < tol: 
            return -9*y - 6*y
        return -9*y - 6*y*np.log(x)

    def poly_psi_10_xx(self, x, y):
        return 36*(x**2)*y - 8*(y**3)

    def poly_psi_11_xx(self, x, y):
        x = abs(x)
        if x < tol: 
            return -120*(x**2)*y - 160*(y**3) - 240*(y**3) + 720*(x**2)*y
        return -120*(x**2)*y - 160*(y**3)*np.log(x) - 240*(y**3) + 720*(x**2)*y*np.log(x)
    
    def poly_psi(obj, index, x, y):
        if obj.updownsym and index in (7, 8, 9, 10, 11):
            return 0
        elif index == 0:
            return obj.poly_psi_0_xx(x, y)
        elif index == 1:
            return obj.poly_psi_1_xx(x, y)
        elif index == 2:
            return obj.poly_psi_2_xx(x, y)
        elif index == 3:
            return obj.poly_psi_3_xx(x, y)
        elif index == 4:
            return obj.poly_psi_4_xx(x, y)
        elif index == 5:
            return obj.poly_psi_5_xx(x, y)
        elif index == 6:
            return obj.poly_psi_6_xx(x, y)
        else:      
            if index == 7:
                return obj.poly_psi_7_xx(x, y)
            elif index == 8:
                return obj.poly_psi_8_xx(x, y)
            elif index == 9:
                return obj.poly_psi_9_xx(x, y)
            elif index == 10:
                return obj.poly_psi_10_xx(x, y)
            elif index == 11:
                return obj.poly_psi_11_xx(x, y)

class poly_homo_y:
    '''
    1st order partial derivative of the polynomial homogenous 
    functions with respect to y
    '''
    def __init__(self, updownsym):
        self.updownsym = updownsym

    def poly_psi_0_y(self, x, y):
        return 0

    def poly_psi_1_y(self, x, y):
        return 0

    def poly_psi_2_y(self, x, y):
        return 2*y

    def poly_psi_3_y(self, x, y):
        return -8*(x**2)*y

    def poly_psi_4_y(self, x, y):
        x = abs(x)
        if x < tol: 
            return 8*(y**3) - 18*(x**2)*y - 24*(x**2)*y
        return 8*(y**3) - 18*(x**2)*y - 24*(x**2)*y*np.log(x)
    def poly_psi_5_y(self, x, y):
        return -24*(x**4)*y + 32*(x**2)*(y**3)

    def poly_psi_6_y(self, x, y):
        x = abs(x)
        part1 = 48*(y**5) - 560*(x**2)*(y**3) + 150*(x**4)*y
        if x < tol: 
            part2 = 360*(x**4)*y - 480*(x**2)*(y**3)
        else:
            part2 = 360*(x**4)*y*np.log(x) - 480*(x**2)*(y**3)*np.log(x)
        return part1 + part2

    def poly_psi_7_y(self, x, y):
        return 1

    def poly_psi_8_y(self, x, y):
        return x**2

    def poly_psi_9_y(self, x, y):
        x = abs(x)
        if x < tol: 
            return 3*(y**2) - 3*(x**2)
        return 3*(y**2) - 3*(x**2)*np.log(x)

    def poly_psi_10_y(self, x, y):
        return 3*(x**4) - 12*(x**2)*(y**2)

    def poly_psi_11_y(self, x, y):
        x = abs(x)
        if x < tol: 
            return 40*(y**4) - 45*(x**4) - 240*(x**2)*(y**2) + 60*(x**4)
        return 40*(y**4) - 45*(x**4) - 240*(x**2)*(y**2)*np.log(x) + 60*(x**4)*np.log(x)
    
    def poly_psi(obj, index, x, y):
        if obj.updownsym and index in (7, 8, 9, 10, 11):
            return 0
        elif index == 0:
            return obj.poly_psi_0_y(x, y)
        elif index == 1:
            return obj.poly_psi_1_y(x, y)
        elif index == 2:
            return obj.poly_psi_2_y(x, y)
        elif index == 3:
            return obj.poly_psi_3_y(x, y)
        elif index == 4:
            return obj.poly_psi_4_y(x, y)
        elif index == 5:
            return obj.poly_psi_5_y(x, y)
        elif index == 6:
            return obj.poly_psi_6_y(x, y)
        else:
            if index == 7:
                return obj.poly_psi_7_y(x, y)
            elif index == 8:
                return obj.poly_psi_8_y(x, y)
            elif index == 9:
                return obj.poly_psi_9_y(x, y)
            elif index == 10:
                return obj.poly_psi_10_y(x, y)
            elif index == 11:
                return obj.poly_psi_11_y(x, y)
            
class poly_homo_yy:
    '''
    2nd order partial derivative of the polynomial homogenous 
    functions with respect to y
    '''
    def __init__(self, updownsym):
        self.updownsym = updownsym

    def poly_psi_0_yy(self, x, y):
        return 0

    def poly_psi_1_yy(self, x, y):
        return 0

    def poly_psi_2_yy(self, x, y):
        return 2

    def poly_psi_3_yy(self, x, y):
        return -8*(x**2)

    def poly_psi_4_yy(self, x, y):
        x = abs(x)
        if x < tol: 
            return 24*(y**2) - 18*(x**2) - 24*(x**2)
        return 24*(y**2) - 18*(x**2) - 24*(x**2)*np.log(x)

    def poly_psi_5_yy(self, x, y):
        return -24*(x**4) + 96*(x**2)*(y**2)

    def poly_psi_6_yy(self, x, y):
        x = abs(x)
        part1 = 240*(y**4) - 1680*(x**2)*(y**2) + 150*(x**4)
        if x < tol:
            part2 = 360*(x**4) - 1440*(x**2)*(y**2)
        else:
            part2 = 360*(x**4)*np.log(x) - 1440*(x**2)*(y**2)*np.log(x)
        return part1+part2

    def poly_psi_7_yy(self, x, y):
        return 0

    def poly_psi_8_yy(self, x, y):
        return 0

    def poly_psi_9_yy(self, x, y):
        return 6*y

    def poly_psi_10_yy(self, x, y):
        return -24*(x**2)*y

    def poly_psi_11_yy(self, x, y):
        x = abs(x)
        if x < tol: 
            return 160*(y**3) - 480*(x**2)*y
        return 160*(y**3) - 480*(x**2)*y*np.log(x)
    
    def poly_psi(obj, index, x, y):
        if obj.updownsym and index in (7, 8, 9, 10, 11):
            return 0
        if index == 0:
            return obj.poly_psi_0_yy(x, y)
        elif index == 1:
            return obj.poly_psi_1_yy(x, y)
        elif index == 2:
            return obj.poly_psi_2_yy(x, y)
        elif index == 3:
            return obj.poly_psi_3_yy(x, y)
        elif index == 4:
            return obj.poly_psi_4_yy(x, y)
        elif index == 5:
            return obj.poly_psi_5_yy(x, y)
        elif index == 6:
            return obj.poly_psi_6_yy(x, y)
        else:
            if index == 7:
                return obj.poly_psi_7_yy(x, y)
            elif index == 8:
                return obj.poly_psi_8_yy(x, y)
            elif index == 9:
                return obj.poly_psi_9_yy(x, y)
            elif index == 10:
                return obj.poly_psi_10_yy(x, y)
            elif index == 11:
                return obj.poly_psi_11_yy(x, y)
            
### === Particular Solution
class parti_solu:
    def __init__(self, word):
        '''
        0 - NONE
        1 - X
        2 - XX
        3 - Y
        4 - YY
        '''
        self.word = word
        self.word_dict = {
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
    
    def particular_sol(self, x, y, A):
        logx = safe_log(x)
        return (x**4)/8 + A*((1/2)*(x**2)*logx - (x**4)/8)
    
    def particular_sol_x(self, x, y, A):
        logx = safe_log(x)
        return (1/2)*x**3 + A*(x*logx + (x/2) - (1/2)*x**3)
    
    def particular_sol_xx(self, x, y, A):
        logx = safe_log(x)
        return (3/2)*x**2 + A*(logx - (3/2)*x**2 + (3/2))
    
    def particular_sol_y(self, x, y, A): 
        return 0
    
    def particular_sol_yy(self, x, y, A):
        return 0
    
    def boundary_condition(self, x, y, A):
        value = self.word_dict[self.word]
        if value == 0:
            return self.particular_sol(x, y, A)
        elif value == 1:
            return self.particular_sol_x(x, y, A)
        elif value == 2: 
            return self.particular_sol_xx(x, y, A)
        elif value == 3:
            return self.particular_sol_y(x, y, A)
        elif value == 4:
            return self.particular_sol_yy(x, y, A)

    def bc_boundary_condition(self, x, y, A):
        value = self.word_dict[self.word]
        if value in (0, 1, 3):
            return 0
        elif value == 2:
            return self.particular_sol_y(x, y, A)
        elif value == 4:
            return self.particular_sol_x(x, y, A)


#######################################################################################################
#######################################################################################################
#################################                 FUNCTIONS                ############################
#######################################################################################################
#######################################################################################################

def delta_tri(alpha):                       # sin(alpha) = delta (triangularity)
    return np.sin(alpha)

def alp_delt(delta):
    return np.arcsin(delta)

### === Curvature Constants 
"""
a = alpha
e = epsilon
k = kappa 
"""
def co_N1(a, e, k):               # tau = 0
    return (-(1+a)**2)/(e*k**2)

def co_N2(a, e, k):               # tau = pi
    return ((1-a)**2)/(e*k**2)

def co_N3(a, e, k):               # tau = pi/2 
    return (-k)/(e*(np.cos(a))**2)

# For FRCs specifically 
def co_N1_FRC(k):
    return (-2/k**2)

def co_N2_FRC(k):
    return (2/k**2)

def co_N3_FRC(k): 
    return (-k/4)

def find_N(a, e, k):
    params = [np.cos(a), e, k]
    N = []

    if any(abs(entry) < 1e-9 for entry in params):
        N.append(co_N1_FRC(k))
        N.append(co_N2_FRC(k))
        N.append(co_N3_FRC(k))
    else:
        N.append(co_N1(a, e, k))
        N.append(co_N2(a, e, k))
        N.append(co_N3(a, e, k))
    return N

### === Calculations
def find_inverse_matrix(mat_d, homo_d, bc_homo_d, d_parti_d, word_d, all_pivots):
    '''
    Finding the inverse matrix. If inverse matrix DNE, this method finds an invertible 
    matrix from the given matrix, mat_d.

    homo_d = homogeneous dictionary
    bc_homo_d = bc homogeneous dictionary
    d_parti_d = difference particular dictionary
    word_d = word dictionary
    all_pivots = set of indices for the base dictionaries
    '''
    if len(mat_d) != len(mat_d[0]) or np.linalg.det(mat_d) == 0:
        rref_col = []
        rref_row = []
        invertible_matrix = []

        mat_d_trans = [[mat_d[j][i] for j in range(len(mat_d))] for i in range(len(mat_d[0]))]
        
        # converts from NumPy to SP
        sp_matrix_col = sp.Matrix(mat_d)
        sp_matrix_row = sp.Matrix(mat_d_trans)
        RREF_matrix_col, p_cols = sp_matrix_col.rref()
        RREF_matrix_row, p_rows = sp_matrix_row.rref()
        # making RREF matrix into a NP
        index = 0
        for index in range(len(p_cols)):
            lst = RREF_matrix_col[index:index+len(all_pivots)]
            index += len(all_pivots)
            rref_col.append(lst)
        index = 0
        for index in range(len(p_rows)):
            lst = RREF_matrix_row[index:index+len(all_pivots)]
            index += len(all_pivots)
            rref_row.append(lst)
    
        # Removing column values from dict if column is dependent
        for key, lst in homo_d.items():
            to_del_col = sorted(list(all_pivots - set(p_cols)), reverse=True)
            for value in to_del_col:
                del homo_d[key][value]
                to_del_col = [v - 1 for v in to_del_col]

        for key, lst in bc_homo_d.items():
            to_del_col = sorted(list(all_pivots - set(p_cols)), reverse=True)
            for value in to_del_col:
                del bc_homo_d[key][value]
                to_del_col = [v - 1 for v in to_del_col]
        # Removing rows from dict if row is dependent
        to_del_row = sorted(list(all_pivots - set(p_rows)), reverse=True)
        for value in to_del_row:
            if value in homo_d.keys():
                del homo_d[value]
                del bc_homo_d[value]
                del d_parti_d[value]
                to_del_row = [v - 1 for v in to_del_row]
        
        # Adding independent rows to invertible_matrix
        for row_i in range(len(mat_d)):
            if row_i in p_rows:
                invertible_matrix.append(mat_d[row_i])

        # Deleting dependent columns from invertible matrix
        for row_i in range(len(invertible_matrix)):
            to_del_col = sorted(list(all_pivots - set(p_cols)), reverse=True)
            for col_i in to_del_col:
                del invertible_matrix[row_i][col_i]
                to_del_col = [v - 1 for v in to_del_col]

        # If still not invertible:
        if len(invertible_matrix) != len(invertible_matrix[0]) or np.linalg.det(invertible_matrix) == 0:
            return None, None
        else: 
            inverse_matrix = np.linalg.inv(invertible_matrix)
            col = p_cols
    else:
        inverse_matrix = np.linalg.inv(mat_d)
        col = all_pivots
    return inverse_matrix, col


def flux(x, y, Ae, c_dict, word_dict, updownsym):
    """
    x = float
    y = float
    Ae = float
    c_dict = constant dictionary
    word_dict = word dictionary
    updownsym = boolean value
    """
    obj_h = poly_homo(updownsym)
    var_parti = parti_solu(word_dict)
    sum = var_parti.particular_sol(x, y, Ae)
    for i in c_dict:
        val = obj_h.poly_psi(i, x, y)
        sum += val*c_dict[i]
    return sum


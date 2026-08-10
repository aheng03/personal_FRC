# Author: Adelina Hengyucius
# Grad_Shafranov User Interface

### --- LIBRARIES 
import sys
import tkinter as tk

#######################################################################################################
#######################################################################################################
#################################                 USER INPUT               ############################
#######################################################################################################
#######################################################################################################

defaultFont = ("Helvetica", 15)
variables = {}

### === Functions
def switch(button):
    if button.is_on:
        button.config(image=off)
        button.is_on = False
    else:
        button.config(image=on)
        button.is_on = True

def create_toggle_button(parent, row, col):
    btn = tk.Button(parent, image=off, bd=0)
    btn.is_on = False
    btn.config(command=lambda: switch(btn))
    btn.grid(row=row, column=col)
    return btn

def create_text_entry(parent, row, col):
    ent = tk.Entry(parent)
    ent.grid(row=row, column= col)
    return ent

def submit():
    variables['elong'] = e_elong.get()
    variables['delt'] = e_delt.get()
    variables['A'] = e_A.get()
    variables['Xs'] = e_Xs.get()
    variables['eps'] = e_eps.get()
    variables['axisSymmetric'] = b_axis.is_on
    variables['smooth'] = b_smooth.is_on
    variables['double_null'] = b_doubled.is_on

    root.destroy()
      

### === Window Settings
root = tk.Tk()
root.title("Plasma Parameters")

### === Images 
on = tk.PhotoImage(file = "Grad_Shaf\images\on.png")
on = on.subsample(3,3)
off = tk.PhotoImage(file = "Grad_Shaf\images\off.png")
off = off.subsample(3, 3)


### === Labels
l_axis = tk.Label(root, text="Up-Down Symmetry", 
                  font = defaultFont)
l_axis.grid(row = 0, column= 1)

l_smooth = tk.Label(root, text="Smooth", 
                    font = defaultFont)
l_smooth.grid(row = 0, column= 2)

l_doubled = tk.Label(root, text="Double Null Divertor", 
                     font = defaultFont)
l_doubled.grid(row = 0, column= 3)

l_elong = tk.Label(root, text = "Elongation =",
                 font = defaultFont)
l_elong.grid(row=2, column = 1)

l_delt = tk.Label(root, text = "Delta =",
                  font = defaultFont)
l_delt.grid(row=3, column = 1)

l_A = tk.Label(root, text = "A =",
               font = defaultFont)
l_A.grid(row=4, column = 1)

l_eps = tk.Label(root, text = "Epsilon =",
                 font = defaultFont)
l_eps.grid(row=5, column = 1)

l_Xs = tk.Label(root, text = 'Xs =', 
                font = defaultFont)
l_Xs.grid(row=6, column = 1)

### === Executables
# Toggle Buttons
b_axis = create_toggle_button(root, 1, 1)
b_smooth = create_toggle_button(root, 1, 2)
b_doubled = create_toggle_button(root, 1, 3)

# Buttons
submit = tk.Button(root, text="Enter", command = submit)
submit.grid(row = 5, column= 3)

### === Text Entries
e_elong = create_text_entry(root, 2, 2)
e_delt = create_text_entry(root, 3, 2)
e_A = create_text_entry(root, 4, 2)
e_eps = create_text_entry(root, 5, 2)
e_Xs = create_text_entry(root, 6, 2)

### === Window Execution
root.mainloop()

if any(v == '' for v in variables.values()) or variables == dict():
    print("ERROR. INPUT INVALID")
    sys.exit()

E = float(variables['elong'])
delt = float(variables['delt'])
A = float(variables['A'])
eps = float(variables['eps'])
Xs = float(variables['Xs'])
axisSymmetric = variables['axisSymmetric']
smooth = variables['smooth']
double_null = variables['double_null']

if double_null and smooth:
    print("ERROR! double_null and smooth cannot be true simultaneously. Exiting program...")
    sys.exit()
if double_null and axisSymmetric is not True:
    print("ERROR! double_null and axisSymmetric must align. Exiting program...")
    sys.exit()

# Only for Spherical Tokamaks: (?)
equilibrium_beta_limit = True
if equilibrium_beta_limit and axisSymmetric is not True:
    print("ERROR! equilibrium_beta_limit and axisSymmetric must align. Exiting program...")
    if equilibrium_beta_limit and smooth is not True:
        print("ERROR! equilibrium_beta_limit and smooth must align. Exiting program...")

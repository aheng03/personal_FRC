
#FRC 3D Equilibrium Plots         Jeffrey D. Contri; jcontri@uw.edu           2025_04_30 - 2025_06_23
#################################################
#####               IMPORTS                 #####
#################################################
import numpy as np
from FRC_2D import *
from plottingParameters import*
import matplotlib.pyplot as plt





###-------------------------------------------------------------------------------------SURFACE CONTOUR

fig3 = plt.figure(figsize=(2*figXsize, figYsize))
ax3  = fig3.add_subplot(111, projection='3d')
# plot the surface; you can tweak rcount/ccount for resolution
surf = ax3.plot_surface(
    R, Z, Bmag*np.sign(Bz),
    cmap='viridis',
    rcount=100, ccount=100,
    linewidth=0, antialiased=True,
    vmin=vmin,
    vmax=vmax
)

# add a colorbar keyed to the surface
cbar3 = fig3.colorbar(surf, ax=ax3, pad=0.1)
cbar3.set_label(r'$|\vec{B}| \cdot B_z/|B_z|$ $[T]$', fontsize=labelFontSize)

# label axes
ax3.set_xlabel('r [cm]',    fontsize=labelFontSize)
ax3.set_ylabel('z [cm]',    fontsize=labelFontSize)
ax3.set_zlabel('$|\\vec{B}|$ $[T]$', fontsize=labelFontSize)
ax3.set_title('3D Magnetic Field Magnitude', fontsize=titleFontSize)
ax3.set_zlim(vmin, vmax)

#plt.tight_layout()
title_save = "surface_contour"
fig3.savefig(title_save + ".png", dpi=dpi_res)
#plt.show()
print("\tPlot A: " + title_save + ".png saved")



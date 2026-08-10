#FRC 2D Equilibrium Plots         Jeffrey D. Contri; jcontri@uw.edu           2025_04_30 - 2025_06_26
#ψ
#################################################
#####               IMPORTS                 #####
#################################################
from FRC_2D import *
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path



#################################################
#####               FUNCTIONS               #####
#################################################
###-----CONVERTS AN INTEGER TO A LETTER
def number_to_letter(n):
    """1→'a', 2→'b', … 26→'z'. Raises ValueError otherwise."""
    if 1 <= n <= 26:
        return chr(ord('A') + n - 1)            #'a' for lowercase; 'A' for uppercase
    raise ValueError(f"Out of range: {n}")





t0code = time.perf_counter()
print("\n\nFRC 2D Plots--------------------------------------------------------------------------------")
print("Max ψ = {:.3e} [T*m^2]\t = {:.3f} [T*cm^2]\t   T = kg/(s*C)".format(psi.max(),psi.max()/Lconv**2))
print("Min ψ = {:.3e} [T*m^2]\t = {:.3f} [T*cm^2]\t   T = kg/(s*C)".format(psi.min(),psi.min()/Lconv**2))
print("NOTE:\tThere should be no negative flux. It exists between the separatrices of the ψ=0 contours of the internal and external solutions. Steinhauer remarks, The internal and external solutions are matched in an approximate sense. See DOI: 10.1063/1.859219 for reference. All negative psi values:")
print(neg_vals)
print("\n")



#################################################
#####               PLOTTING                #####
#################################################
plot_idx = 0
figXsize = figXsize/2
numLevs = 25
vmax = Bw*1.2
vmin = -vmax


r0_sep, z0_sep = get_flux_contours(psi, R, Z, 0)       #separatrix coordinates for the combined flux [m]
r0int, z0int = get_flux_contours(psi_int, R, Z, 0)     #separatrix coordinates for the internal flux [m]
r0ext, z0ext = get_flux_contours(psi_ext, R, Z, 0)     #separatrix coordinates for the external flux [m]

r0int = r0int/Lconv
z0int = z0int/Lconv

r0ext = r0ext/Lconv
z0ext = z0ext/Lconv


###-----------------------------------------------------------------------------------------FLUX CONTOUR
plot_idx = plot_idx + 1
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')  # equal scaling on both axes

levels = Lconv**2 * np.arange(-0.1, 1.2, 0.1)       #defines contour levels at ψ = 0.0, 0.1, 0.2, …, 1.0
#levels = np.linspace(psi.min(), psi.max(), numLevs)  


# draw thin black contour lines at every 0.1
cs = ax.contour(R/Lconv, Z/Lconv, psi,
                levels=levels,
                colors='k',
                linewidths=0.5)
"""
# fill between contours with a colormap
cf = ax.contourf(R/Lconv, Z/Lconv, psi,
                 levels=levels,
                 cmap='viridis')
"""

# highlight ψ=0 and ψ=1 with thicker lines
ax.contour(R/Lconv, Z/Lconv, psi,
           levels=[0.0, 1.0*Lconv**2],
           colors='k',
           linewidths=1)

# label each contour line with its ψ value
ax.clabel(cs,
          fmt="%1.e",
          inline=True,
          fontsize=8)

ax.set_xlabel('r [cm]')
ax.set_ylabel('z [cm]')
ax.set_title(r'$\psi(r,z)$ contours')
title_save = "flux_contour"
plt.tight_layout()
plt.savefig(number_to_letter(plot_idx) + "_" + title_save + ".png", dpi=dpi_res)
#plt.show()
print("\tPlot A: " + title_save + ".png saved")


'''

###-----------------------------------------------------------------(ARROW GRAPH) MAGNETIC FIELD VECTORS
plot_idx = plot_idx + 1
fig, ax = plt.subplots(figsize=(figXsize, figYsize))
ax.set_aspect('equal')                    # equal scaling on r & z
n = 20     # draw arrows at every nth point to avoid crowding:
Q = ax.quiver(
    R[::n,::n]/Lconv, Z[::n,::n]/Lconv,               # arrow positions
    Br[::n,::n], Bz[::n,::n],             # arrow components
    pivot='tail',                          # draw arrows centered on grid points
    scale=50,                             # tweak to make arrow lengths legible
    headwidth=6, headlength=6, headaxislength=4
)

ax.set_xlabel('r [cm]')
ax.set_ylabel('z [cm]')
ax.set_title('Magnetic Field $\\vec{B}(r,z)$')
# optional: add a key showing the arrow scale

plt.tight_layout()
title_save = "Arrows"
plt.savefig(number_to_letter(plot_idx) + "_" + title_save + ".png", dpi=dpi_res)
#plt.show()
print("\tPlot B: " + title_save + ".png saved")



###--------------------------------------------------------------------------ARROWS BETWEEN SEPARATRICES
plot_idx = plot_idx + 1

# 1) build closed Path objects for the two flux‐loops
verts_int  = np.vstack((r0int, z0int)).T
verts_ext = np.vstack((r0ext, z0ext)).T

# ensure each polygon is closed
if not np.allclose(verts_int[0], verts_int[-1]):
    verts_int = np.vstack((verts_int, verts_int[0]))
if not np.allclose(verts_ext[0], verts_ext[-1]):
    verts_ext = np.vstack((verts_ext, verts_ext[0]))
path_int  = Path(verts_int)
path_ext = Path(verts_ext)

# 2) downsample your grid by factor n before doing any point-in-polygon tests
n    = 10
Ri   = R[::n, ::n]/Lconv
Zi   = Z[::n, ::n]/Lconv
Bri  = Br[::n, ::n]
Bzi  = Bz[::n, ::n]

# 3) test inclusion on the small set of points only
pts        = np.vstack((Ri.flatten(), Zi.flatten())).T
inside_int  = path_int.contains_points(pts).reshape(Ri.shape)
inside_ext = path_ext.contains_points(pts).reshape(Ri.shape)

# 4) “between” means XOR of the two masks
#mask       = np.logical_xor(inside_int, inside_ext)
mask = np.logical_and( inside_ext, np.logical_not(inside_int) )

# 5) mask out B everywhere but that band
Bri_masked = np.ma.masked_where(~mask, Bri)
Bzi_masked = np.ma.masked_where(~mask, Bzi)

# 6) plot
fig, ax = plt.subplots(figsize=(figXsize, figYsize))
ax.set_aspect('equal')

ax.plot(r0int, z0int, '-', color='red', label='Internal', zorder=1)
ax.plot(r0ext, z0ext, '-', color='blue', label='External', zorder=1)

ax.quiver(Ri, Zi,
          Bri_masked, Bzi_masked,
          pivot='tail',
          scale=50,
          headwidth=6, headlength=6, headaxislength=4)

ax.set_xlabel('r [cm]')
ax.set_ylabel('z [cm]')
ax.set_xlim(0/Lconv, Rc/Lconv)
ax.set_ylim(Z0/3/Lconv, 0.4*h/Lconv)
ax.set_title('Magnetic Field $\\vec B(r,z)$ between $\\psi_{int}=0$ and $\\psi_{ext}=0$')
ax.legend()
title_save = "arrows between contours"
plt.tight_layout()
plt.savefig(number_to_letter(plot_idx) + "_" + title_save + ".png", dpi=dpi_res)
print("\tPlot C: " + title_save + ".png saved")



###------------------------------------------------------------------------------------------STREAMPLOT
plot_idx = plot_idx + 1
fig, ax = plt.subplots(figsize=(figXsize, figYsize))
ax.set_aspect('equal')
strm = ax.streamplot(
    r/Lconv, z/Lconv,                                # your 1D coordinate vectors
    Br.T, Bz.T,                          # note the transpose for streamplot
    density=(0.5, 1.0),                        # controls number of streamlines
    color='k',
    linewidth=1.0,
    arrowstyle='->'
)
ax.set_xlabel('r [cm]')
ax.set_ylabel('z [cm]')
ax.set_title('Magnetic Field Lines')
title_save = "streamplot"
plt.savefig(number_to_letter(plot_idx) + "_" + title_save + ".png", dpi=dpi_res)
#plt.show()
print("\tPlot D: " + title_save + ".png saved")




###------------------------------------------------------------------------------MIDPLANE PROFILE CHECK
plot_idx = plot_idx + 1
mid_z = np.argmin(np.abs(z))
mid_r = int(len(r)/2)

rmid = r[mid_r:]
Bmid = Bz[mid_r:, mid_z]

# ─── Draw a NEW 2D figure for your mid-plane cut ──────────────────────────
fig4, ax4 = plt.subplots(figsize=(1.5*figXsize, figYsize))
ax4.plot(rmid/Lconv, Bmid, color='k', linewidth=lineWidth)      # plot your midplane data
ax4.set_xlabel('r [cm]', fontsize=labelFontSize)
ax4.set_ylabel(r'$|\vec{B}| = B_z$ $[T]$', fontsize=labelFontSize)
ax4.set_title('Mid-plane Magnetic Field Strength', fontsize=titleFontSize)


#plt.tight_layout()
title_save = "midplane_check"
fig4.savefig(number_to_letter(plot_idx) + "_" + title_save + ".png", dpi=dpi_res)
#plt.show()
print("\tPlot E: " + title_save + ".png saved")





###------------------------------------------------------------------------------------CONTOUR + ARROWS
plot_idx = plot_idx + 1
fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.set_aspect('equal')

# 1) compute the magnitude on the full grid
Bmag = np.sqrt(Br**2 + Bz**2)

# 2) draw a filled contour of |B|
cf = ax2.contourf(R/Lconv, Z/Lconv, Bmag*np.sign(Bz), levels=50, cmap='viridis')

# 3) overlay a sparse quiver showing direction
n = 25
Q2 = ax2.quiver(
    R[::n, ::n]/Lconv, Z[::n, ::n]/Lconv,          # positions
    Br[::n, ::n], Bz[::n, ::n],        # vector components
    pivot='mid', scale=50,
)

# 4) add a colorbar for the contour
cbar = fig2.colorbar(cf, ax=ax2, pad=0.1)
cbar.set_label(r'$|\vec{B}|$ $[T]$', fontsize=labelFontSize)

# 5) labels and title
ax2.set_xlabel('r [cm]')
ax2.set_ylabel('z [cm]')
ax2.set_title('Magnetic Field Magnitude and Direction')

plt.tight_layout()
title_save = "contour_arrows2"
plt.savefig(number_to_letter(plot_idx) + "_" + title_save + ".png", dpi=dpi_res)
#plt.show()
print("\tPlot F: " + title_save + ".png saved")





###-------------------------------------------------------------------------------------ZERO FLUX GIVEN
plot_idx = plot_idx + 1
r0_int, z0_int = get_flux_contours(psi_int, R, Z, 0)
r0_ext, z0_ext = get_flux_contours(psi_ext, R, Z, 0)

r0_int = r0_int/Lconv
z0_int = z0_int/Lconv

r0_ext = r0_ext/Lconv
z0_ext = z0_ext/Lconv

r0_comb = r0_comb/Lconv
z0_comb = z0_comb/Lconv


fig, axes = plt.subplots(1, 4, figsize=(16, 8), constrained_layout=True)

# 1) internal separatrix
axes[0].plot(r0_int, z0_int, 'r-', linewidth=2)
title0 = "$\\psi_{int}(r,z) = 0$ Contour"
axes[0].set_title(title0)
axes[0].set_xlabel('r [cm]')
axes[0].set_ylabel('z [cm]')
axes[0].set_aspect('equal')

# 2) external separatrix
axes[1].plot(r0_ext, z0_ext, 'b-', linewidth=2)
title1 = "$\\psi_{ext}(r,z) = 0$ Contour"
axes[1].set_title(title1)
axes[1].set_xlabel('r [cm]')
axes[1].set_ylabel('z [cm]')
axes[1].set_aspect('equal')

# 3) overlay both
axes[2].plot(r0_int, z0_int, 'r-', linewidth=2, label='internal')
axes[2].plot(r0_ext, z0_ext, 'b--', linewidth=2, label='external')
axes[2].plot(np.abs(r0_comb), np.abs(z0_comb), 'c:', linewidth=2, label='combined')
axes[2].plot(np.abs(r0_comb), z0_comb, 'c:', linewidth=2)
axes[2].plot(r0_comb, np.abs(z0_comb), 'c:', linewidth=2)
axes[2].plot(r0_comb, z0_comb, 'c:', linewidth=2)
title2 = "Overlay $\\psi = 0$ Contours"
axes[2].set_title(title2)
axes[2].set_xlabel('r [cm]')
axes[2].set_ylabel('z [cm]')
axes[2].legend(loc='best')
axes[2].set_aspect('equal')

# 4) discrepancy amplification
axes[3].plot(r0_int, z0_int, 'r-', linewidth=2, label='internal')
axes[3].plot(r0_ext, z0_ext, 'b--', linewidth=2, label='external')
axes[3].plot(np.abs(r0_comb), np.abs(z0_comb), 'c:', linewidth=2, label='combined')
axes[3].plot(r0_comb, np.abs(z0_comb), 'c:', linewidth=2)
title3 = "Refined Overlay"
axes[3].set_title(title3)
axes[3].set_xlabel('r [cm]')
axes[3].set_ylabel('z [cm]')
axes[3].set_ylim(Z0/3/Lconv, 0.4*h/Lconv)
axes[3].legend(loc='best')
axes[3].set_aspect('equal')
axes[3].xaxis.set_minor_locator(AutoMinorLocator(4))
axes[3].yaxis.set_minor_locator(AutoMinorLocator(4))
axes[3].grid(which='major', color='k', linestyle='-', linewidth=0.8)
axes[3].grid(which='minor', color='gray', linestyle=':', linewidth=0.5)


title_save = "Zero_Flux_Contour"
fig.savefig(number_to_letter(plot_idx) + "_" + title_save + ".png", dpi=dpi_res)
print("\tPlot G: " + title_save + ".png saved")





###-----------------------------------------------------------------------FIVE-PANEL LANDSCAPE OVERVIEW
plot_idx = plot_idx + 1
levels = Lconv**2 * np.arange(-0.1, 1.5, 0.1)       #defines contour levels at ψ = 0.0, 0.1, 0.2, …, 1.0
fig = plt.figure(figsize=(24, 12))
fig.subplots_adjust(left=0.025, right=0.975, top=0.975, bottom=0.025)   #figure margins


# 1) Upper‐Left: ψ contours & streamlines with +z→left, +r→up
ax1 = fig.add_axes([0.05, 0.55, 0.50, 0.46])    # [left, bottom, width, height]

# mask half‐domains
psi_pos = np.where(R > 0, psi, np.nan)
Br_neg  = np.where(R < 0, Br,  np.nan)
Bz_neg  = np.where(R < 0, Bz,  np.nan)

# define physical axes
x1 = z/Lconv           # x = z
y1 = r/Lconv           # y = r

# vector components on (x=z, y=r)
U1 = Bz_neg
V1 = Br_neg

# draw filled ψ‐contours then streamlines
ax1.contourf(x1, y1, psi_pos,
             levels=levels, cmap='viridis', origin='lower', zorder=1)
ax1.streamplot(x1, y1, U1, V1,
               density=0.5, color='k',
               linewidth=0.8, arrowstyle='->', zorder=2)

# flip x‐axis so positive z appears on the left
ax1.invert_xaxis()

# labels, title, and panel number
ax1.set_xlabel('z [cm]')
ax1.set_ylabel('r [cm]')
ax1.set_title('1: Contours & Streamlines')
ax1.text(0.02, 0.98, '1',
         transform=ax1.transAxes,
         fontsize=18, fontweight='bold', va='top')
ax1.set_aspect('equal')


# 2) Lower‐Left: quiver, reduced density
ax2 = fig.add_axes([0.0, 0.05, 0.25, 0.49])
iz0 = np.searchsorted(z, 0)
iz1 = np.searchsorted(z, 3*h/2)
ir0 = np.searchsorted(r, -Rw)
ir1 = np.searchsorted(r, Rw)
R2, Z2 = R[ir0:ir1, iz0:iz1], Z[ir0:ir1, iz0:iz1]
Br2, Bz2 = Br[ir0:ir1, iz0:iz1], Bz[ir0:ir1, iz0:iz1]
R2 = R2/Lconv
Z2 = Z2/Lconv

step = 20                                               #high step = fewer arrows

ax2.quiver(
    R2[::step, ::step], Z2[::step, ::step],
    Br2[::step, ::step], Bz2[::step, ::step],
    pivot='tail',
    scale=75,               # ↑ increase this to make arrows shorter
    scale_units='xy',       # interpret “scale” in data‐units
    angles='xy',            # match your grid orientation
    width=0.005,            # thinner shafts
    headwidth=3,            # modest arrow‐head size
    headlength=3
)

ax2.set_xlabel('r [cm]'); ax2.set_ylabel('z [cm]')
ax2.set_title('2: Magnetic Field Direction')
ax2.text(0.925, 0.05, '2',
         transform=ax2.transAxes,
         fontsize=18, fontweight='bold', va='top')
ax2.set_aspect('equal')



# 3) Lower‐Middle: rotated like plot 1 (r up, z left), highlight ψ≈0 & ψ≈1
ax3 = fig.add_axes([0.25, 0.20, 0.3, 0.5])
ir = np.where((r >= 0) & (r <= Rw))[0]
iz = np.where((z >= 0) & (z <= h/2))[0]
R3, Z3 = R[np.ix_(ir, iz)], Z[np.ix_(ir, iz)]
R3 = R3*Lconv
Z3 = Z3*Lconv
psi3    = psi[np.ix_(ir, iz)]

# build mesh in the rotated coordinates
X3, Y3 = np.meshgrid(z[iz], r[ir])
X3 = X3/Lconv
Y3 = Y3/Lconv

ax3.invert_xaxis()

# find the two level‐values nearest to 0.0 and 1.0
levs_arr = np.array(levels)
l0 = levs_arr[np.abs(levs_arr - 0.0).argmin()]
l1 = levs_arr[np.abs(levs_arr - 1.0).argmin()]

# draw all contours faintly in grey
ax3.contour(X3, Y3, psi3,
            levels=levels,
            colors='gray',
            linewidths=0.5)

# overlay the ψ≈0 line in red and ψ≈1 line in blue
ax3.contour(X3, Y3, psi3,
            levels=[l0],
            colors='red',
            linewidths=2)
ax3.contour(X3, Y3, psi3,
            levels=[l1],
            colors='blue',
            linewidths=2)

ax3.set_xlabel('z [cm]'); ax3.set_ylabel('r [cm]')
ax3.set_title('3: Highlight ψ≈0 (red), ψ≈1 (blue)')
ax3.text(0.95, 0.975, '3',
         transform=ax3.transAxes,
         fontsize=18, fontweight='bold', va='top')
ax3.set_aspect('equal')

summary_text = summary_text  
ax3.text(0.1, -0.3, summary_text,
         transform=ax3.transAxes,
         ha='left', va='top',
         fontsize=10, wrap=True)



# 4) Full‐domain ψ contours (vertical panel)
ax4 = fig.add_axes([0.55, 0.05, 0.20, 0.90])
cs4 = ax4.contour(R/Lconv, Z/Lconv, psi, levels=levels,
                  colors='k', linewidths=1)
#ax4.clabel(cs4, fmt='%.1f', inline=True, fontsize=8)
ax4.set_xlabel('r [cm]'); ax4.set_ylabel('z [cm]')
ax4.set_title('4: Full‐Domain ψ Contours')
ax4.text(0.02, 0.98, '4',
         transform=ax4.transAxes,
         fontsize=18, fontweight='bold', va='top')
ax4.set_aspect('equal')



# 5) Far‐Right: color contour of Bmag * sign(Bz)
ax5 = fig.add_axes([0.75, 0.05, 0.18, 0.90])
Bmag = np.sqrt(Br**2 + Bz**2)
Bsign = Bmag * np.sign(Bz)
cf5 = ax5.contourf(R/Lconv, Z/Lconv, Bsign, levels=50, cmap='viridis')
cbar5 = fig.colorbar(cf5, ax=ax5, pad=0.05)
cbar5.set_label(r'$|\vec{B}| \cdot B_z/|B_z|$ $[T]$', fontsize=labelFontSize)
ax5.set_xlabel('r [cm]')
ax5.set_ylabel('z [cm]')
ax5.set_title('5: Magnetic Field')
ax5.text(0.02, 0.98, '5',
         transform=ax5.transAxes,
         fontsize=18, fontweight='bold', va='top')
ax5.set_aspect('equal')

#plt.tight_layout()
title_save = "five_panel_update"
fig.savefig(number_to_letter(plot_idx) + "_" + title_save + ".png", dpi=dpi_res)
#plt.show()
print("\tPlot H: " + title_save + ".png saved")






###---------------------------------------------------------------------------FLUX CONTOUR WITH MINIMUM
plot_idx = plot_idx + 1
# 1) Upper‐Left: ψ contours & streamlines with +z→left, +r→up
fig1, ax1 = plt.subplots(figsize=(10,5))

# mask half‐domains
psi_pos = np.where(R > 0, psi, np.nan)
Br_neg  = np.where(R < 0, Br,  np.nan)
Bz_neg  = np.where(R < 0, Bz,  np.nan)

# define physical axes
x1 = z/Lconv           # x = z
y1 = r/Lconv           # y = r

# vector components on (x=z, y=r)
U1 = Bz_neg
V1 = Br_neg

# 1) draw and label all ψ‐contours in black
CS = ax1.contour(
    x1, y1, psi,
    levels=levels,
    colors='k',
    linewidths=1,
    linestyles='solid'
)
ax1.clabel(
    CS,
    fmt='%.e',
    inline=True,
    fontsize=8,
    colors='k'
)

# 2) overlay the ψ=1 separatrix in blue, thicker
CS1 = ax1.contour(
    x1, y1, psi,
    levels=[1.0],
    colors='blue',
    linewidths=2,
    linestyles='solid'
)
ax1.clabel(
    CS1,
    fmt={1.0: r'$\psi=1$'},
    inline=True,
    fontsize=10,
    colors='blue'
)

# 3a) overlay the ψ=0 separatrix in red
CS0 = ax1.contour(
    x1, y1, psi,
    levels=[0.0],
    colors='red',
    linewidths=1,
    linestyles='solid',
    zorder=3
)
# optional: label that red line as ψ=0
ax1.clabel(
    CS0,
    fmt={0.0: r'$\psi=0$'},
    inline=True,
    fontsize=10,
    colors='red'
)


# 3) find and mark the global minimum of psi
min_idx = np.unravel_index(np.argmin(psi), psi.shape)
r_min, z_min = R[min_idx], Z[min_idx]
r_min = r_min/Lconv
z_min = z_min/Lconv

psi_min      = psi[min_idx]
ax1.plot(z_min, -r_min, 'ro', markersize=6)
ax1.text(
    z_min, -r_min,
    f'  min ψ = {psi_min:.2e}',
    color='red', fontsize=10,
    va='bottom', ha='left')


# flip x‐axis so positive z appears on the left
ax1.invert_xaxis()
ax1.grid(True)

# labels, title, and panel number
title_save = 'Full Flux Contour with Minimum_horz'
ax1.set_xlabel('z [cm]')
ax1.set_ylabel('r [cm]')
ax1.set_title(title_save)


plt.tight_layout()
fig1.savefig(number_to_letter(plot_idx) + "_" + title_save + ".png", dpi=dpi_res)
# plt.show()

print("\tPlot I: " + title_save + ".png saved")                                           #Horizontal





######################################################################################################
                                                                                        #Vertical
plot_idx = plot_idx + 1

fig2, ax2 = plt.subplots(figsize=(5, 10))

# 1) draw and label all ψ‐contours in black
CS = ax2.contour(
    R/Lconv, Z/Lconv, psi,
    levels=levels,
    colors='k',
    linewidths=1,
    linestyles='solid'
)
ax2.clabel(
    CS,
    fmt='%1.e',
    inline=True,
    fontsize=8,
    colors='k'
)

# 2) overlay the ψ=1 separatrix in blue, thicker
CS1 = ax2.contour(
    R/Lconv, Z/Lconv, psi,
    levels=[1.0],
    colors='blue',
    linewidths=2,
    linestyles='solid'
)
ax2.clabel(
    CS1,
    fmt={1.0: r'$\psi=1$'},
    inline=True,
    fontsize=10,
    colors='blue'
)

# 3) find and mark the global minimum of psi
min_idx = np.unravel_index(np.argmin(psi), psi.shape)

r_min, z_min = R[min_idx], Z[min_idx]
r_min = r_min/Lconv
z_min = z_min/Lconv

psi_min      = psi[min_idx]
ax2.plot(r_min, z_min, 'ro', markersize=6)
ax2.text(
    r_min, z_min,
    f'  min ψ = {psi_min:.2e}',
    color='red', fontsize=10,
    va='bottom', ha='left'
)

# 4) finalize
ax2.set_xlabel('r [cm]')
ax2.set_ylabel('z [cm]')
ax2.set_title('Full‐Domain ψ Contours')
ax2.set_aspect('equal')
ax2.grid(True)

title_save = 'Full Flux Contour with Minimum_vert'
fig2.savefig(number_to_letter(plot_idx) + "_" + title_save + ".png", dpi=dpi_res)
# plt.show()
print("\tPlot J: " + title_save + ".png saved")






###----------------------------------------------------------------------------------ARROWS TRIPLE PLOT
plot_idx = plot_idx + 1

fig, ax = plt.subplots(1, 3, figsize=(12, 8), constrained_layout=True)
ax[0].set_position([0.05, 0.1, 0.25, 0.8])
ax[1].set_position([0.35, 0.1, 0.25, 0.8])
ax[2].set_position([0.7, 0.1, 0.25, 0.8])

# 1) Full Separatrix
ax[0].plot(r0_int, z0_int, 'r-', linewidth=2, label='internal')
ax[0].plot(r0_ext, z0_ext, 'b--', linewidth=2, label='external')
title2 = "Overlay $\\psi = 0$ Contours"
ax[0].set_title(title2)
ax[0].set_xlabel('r [cm]')
ax[0].set_ylabel('z [cm]')
ax[0].legend(loc='best')
ax[0].set_aspect('equal')



# 2) Arrows Inside
verts_int  = np.vstack((r0int, z0int)).T
verts_ext = np.vstack((r0ext, z0ext)).T
# ensure each polygon is closed
if not np.allclose(verts_int[0], verts_int[-1]):
    verts_int = np.vstack((verts_int, verts_int[0]))
if not np.allclose(verts_ext[0], verts_ext[-1]):
    verts_ext = np.vstack((verts_ext, verts_ext[0]))
path_int  = Path(verts_int)
path_ext = Path(verts_ext)

n    = 5
Ri   = R[::n, ::n]/Lconv
Zi   = Z[::n, ::n]/Lconv
Bri  = Br[::n, ::n]
Bzi  = Bz[::n, ::n]

pts        = np.vstack((Ri.flatten(), Zi.flatten())).T
inside_int  = path_int.contains_points(pts).reshape(Ri.shape)
inside_ext = path_ext.contains_points(pts).reshape(Ri.shape)

mask       = np.logical_xor(inside_int, inside_ext)

Bri_masked = np.ma.masked_where(~mask, Bri)
Bzi_masked = np.ma.masked_where(~mask, Bzi)

ax[1].set_aspect('equal')

ax[1].plot(r0int, z0int, '-', color='red', label='Internal', zorder=1)
ax[1].plot(r0ext, z0ext, '-', color='blue', label='External', zorder=1)

ax[1].quiver(Ri, Zi,
          Bri_masked, Bzi_masked,
          pivot='tail',
          scale=50,
          headwidth=6, headlength=6, headaxislength=4)

ax[1].set_xlabel('r [cm]')
ax[1].set_ylabel('z [cm]')
ax[1].set_xlim(0, Rc/Lconv)
ax[1].set_ylim(Z0/3/Lconv, 5/4*Z0/Lconv)
ax[1].set_title('Magnetic Field $\\vec B(r,z)$ between $\\psi_{int}=0$ and $\\psi_{ext}=0$')
ax[1].legend()
ax[1].set_aspect('equal')


# 3) Arrows Outside
n = 100     # draw arrows at every nth point to avoid crowding:
Q = ax[2].quiver(
    R[::n,::n], Z[::n,::n],               # arrow positions
    Br[::n,::n], Bz[::n,::n],             # arrow components
    pivot='tail',                          # draw arrows centered on grid points
    scale=50,                             # tweak to make arrow lengths legible
    headwidth=5, headlength=5, headaxislength=3
)

ax[2].plot(r0int, z0int, '-', color='red', label='internal', zorder=1)
ax[2].plot(r0ext, z0ext, '-', color='blue', label='external', zorder=1)

ax[2].set_xlabel('r [cm]')
ax[2].set_ylabel('z [cm]')
ax[2].set_title('Magnetic Field $\\vec{B}(r,z)$')
ax[2].set_xlim(0, Rc/Lconv)
ax[2].set_ylim(Z0/3/Lconv, 5/4*Z0/Lconv)
ax[2].set_aspect('equal')

title_save = "Arrows Triple Plot"
fig.savefig(number_to_letter(plot_idx) + "_" + title_save + ".png", dpi=dpi_res)
print("\tPlot K: " + title_save + ".png saved")







###----------------------------------------------------------------------------------divB DOUBLE PLOT
plot_idx = plot_idx + 1
numLevs = 25
#norm_modB = modB / Z0


fig, ax = plt.subplots(1, 2, figsize=(8, 8), constrained_layout=True)
ax[0].set_position([0.1, 0.1, 0.3, 0.8])
ax[1].set_position([0.425, 0.1, 0.5, 0.8])


# 1) Full Separatrix
#ax[0].plot(r0_int, z0_int, 'r-', linewidth=2, label='internal', zorder=2)
#ax[0].plot(r0_ext, z0_ext, 'w--', linewidth=2, label='external', zorder=2)


# 2) Create a masked array that hides those points:
rCirc = 0.05
zCirc = Z0
zStrip = Zd
rStrip = 0.01
rSep1 = Rs-2*dr
rSep2 = Rs+2*dr
zSep = Z0/2
poly = Path(np.column_stack((r0_comb, z0_comb)))
pts = np.column_stack((R.flatten(), Z.flatten()))
inside_poly = poly.contains_points(pts)
inside_poly_mask = inside_poly.reshape(R.shape)
exclude_circle_top    = (R**2 + (Z - zCirc)**2) <= rCirc**2
exclude_circle_bottom = (R**2 + (Z + zCirc)**2) <= rCirc**2
exclude_strip         = (np.abs(R) <= rStrip) & (np.abs(Z) <= zStrip)
exclude_separatrix    = ((np.abs(R) >= rSep1) & (np.abs(R) <= rSep2)) & (np.abs(Z) <= zSep)

#exclude = exclude_circle_top | exclude_circle_bottom | exclude_strip | exclude_separatrix
#exclude = exclude_circle_top | exclude_circle_bottom | exclude_strip
#exclude = exclude_strip | inside_poly_mask | exclude_separatrix
exclude = exclude_strip

divB_masked = np.ma.array(divB_scaled, mask=exclude)
divB_masked = divB_scaled
levels = np.linspace(divB_masked.min(), divB_masked.max(), numLevs)   
#levels = np.linspace(0, 100, numLevs)  
vmin = divB_masked.min()
vmax = divB_masked.max() 
log_norm = LogNorm(vmin=vmin, vmax=vmax)

# 3) Plot using contourf on the masked array:
cf = ax[0].contourf(
    R/Lconv, Z/Lconv, divB_masked,
    levels=levels,
    cmap='viridis'
)

title2 = "Overlay $\\psi = 0$ Contours"
#ax[0].set_title(title2)
ax[0].set_xlabel('r [cm]')
ax[0].set_ylabel('z [cm]')
handles, labels = ax[0].get_legend_handles_labels()
if labels:
    ax[0].legend(handles, labels, loc='best')
ax[0].set_aspect('equal')
#cbar = fig.colorbar(cf, ax=ax[0])
#cbar.set_ticks([])
#cbar.set_ticklabels([])

ax[1].set_aspect('equal')
ax[1].plot(r0int, z0int, '-', linewidth=0.5, color='red', label='internal', zorder=2)
ax[1].plot(r0ext, z0ext, '-', linewidth=0.5, color='white', label='external', zorder=2)

cf = ax[1].contourf(
    R/Lconv, Z/Lconv, divB_masked,
    levels=levels,
    cmap='viridis'
)

# add a colorbar
cbar = fig.colorbar(cf, ax=ax[1])
cbar.set_label(r'$\frac{|\nabla \cdot \vec{B}|}{B_w/R_s}$')

z1 = Z0 / 3 / Lconv
z2 = 5/4 * Z0 / Lconv
ax[1].set_xlabel('r [cm]')
#ax[1].set_ylabel('z [cm]')
ax[1].set_xlim(0, Rc/Lconv)
ax[1].set_ylim(z1, z2)
#ax[1].set_title('Magnetic Field $\\vec B(r,z)$ between $\\psi_{int}=0$ and $\\psi_{ext}=0$')
ax[1].legend()
ax[1].set_aspect('equal')

title_save = "ModB Double Plot"
fig.savefig(number_to_letter(plot_idx) + "_" + title_save + ".png", dpi=dpi_res)
print("\tPlot L: " + title_save + ".png saved")



sys.exit()
######################################
t1code = time.perf_counter()
print(f"Elapsed Time: {t1code-t0code:.2f} [s]")'''
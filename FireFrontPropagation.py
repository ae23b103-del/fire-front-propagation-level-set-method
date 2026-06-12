import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 12
})

RED, BLUE = '#b22222', '#1f77b4'
os.makedirs("figures", exist_ok=True)

# Grid and initial conditions
def make_grid(Lx=100, Ly=100, Nx=100, Ny=100):
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y, x[1]-x[0], y[1]-y[0]

def phi_circle(X, Y, x0, y0, r): return np.sqrt((X-x0)**2 + (Y-y0)**2) - r
def phi_line(X, Y, y0): return Y - y0

def combine(*phis):
    phi = phis[0].copy()
    for p in phis[1:]: phi = np.minimum(phi, p)
    return phi

# Gradient
def central_grad(phi, dx, dy):
    px, py = np.zeros_like(phi), np.zeros_like(phi)
    px[1:-1] = (phi[2:] - phi[:-2]) / (2*dx)
    px[0], px[-1] = (phi[1]-phi[0])/dx, (phi[-1]-phi[-2])/dx
    py[:,1:-1] = (phi[:,2:] - phi[:,:-2]) / (2*dy)
    py[:,0], py[:,-1] = (phi[:,1]-phi[:,0])/dy, (phi[:,-1]-phi[:,-2])/dy
    return px, py

# normal
def get_normal(phi, dx, dy):
    px, py = central_grad(phi, dx, dy)
    mag = np.sqrt(px**2 + py**2)
    mag = np.where(mag < 1e-10, 1e-10, mag)
    return px/mag, py/mag

# Speed functions
def speed_australian(phi, dx, dy, Vx, Vy, r0=0.165, cf=3.24):
    nx, ny = get_normal(phi, dx, dy)
    Vn = Vx*nx + Vy*ny
    Un = np.maximum(r0*(1 + cf*Vn), 1e-6)
    return Un*nx, Un*ny

def speed_mallet(phi, dx, dy, Vx, Vy,
                 r0=0.165, cf=3.24, n=1.5, alpha=0.5):
    nx, ny = get_normal(phi, dx, dy)
    Vm = np.sqrt(Vx**2 + Vy**2) + 1e-10
    cos_theta = np.clip((Vx*nx + Vy*ny)/Vm, -1, 1)
    theta = np.arccos(cos_theta)
    cp = np.maximum(cos_theta, 0)

    head = r0*(1 + cf*np.sqrt(Vm)*cp**n)
    flank = r0*(alpha + (1-alpha)*np.sin(theta))

    Un = np.maximum(
        np.where(np.abs(theta) <= np.pi/2, head, flank),
        1e-6
    )
    return Un*nx, Un*ny

# Minmod limiter
def minmod(a, b):
    return 0.5*(np.sign(a)+np.sign(b))*np.minimum(np.abs(a), np.abs(b))

def compute_rhs(phi, dx, dy, Vx, Vy, speed_fn):
    Ux, Uy = speed_fn(phi, dx, dy, Vx, Vy)
    Nx, Ny = phi.shape
    dphix, dphiy = np.zeros_like(phi), np.zeros_like(phi)

    # X-direction
    for i in range(2, Nx-2):

        db = (phi[i]-phi[i-1])/dx
        df = (phi[i+1]-phi[i])/dx
        slope = minmod(db, df)
        left = db + 0.5*slope
        right = df - 0.5*slope
        dphix[i] = np.where(Ux[i] >= 0, left, right)
    dphix[0], dphix[1] = dphix[2], dphix[2]
    dphix[-1], dphix[-2] = dphix[-3], dphix[-3]

    # Y-direction
    for j in range(2, Ny-2):

        db = (phi[:,j]-phi[:,j-1])/dy
        df = (phi[:,j+1]-phi[:,j])/dy
        slope = minmod(db, df)
        left = db + 0.5*slope
        right = df - 0.5*slope
        dphiy[:,j] = np.where(Uy[:,j] >= 0, left, right)
    dphiy[:,0], dphiy[:,1] = dphiy[:,2], dphiy[:,2]
    dphiy[:,-1], dphiy[:,-2] = dphiy[:,-3], dphiy[:,-3]
    return -(Ux*dphix + Uy*dphiy)

# Time integration
def get_dt(phi, dx, dy, Vx, Vy, speed_fn, CFL=0.45):
    Ux, Uy = speed_fn(phi, dx, dy, Vx, Vy)
    speed = np.max(np.sqrt(Ux**2 + Uy**2))
    return CFL*min(dx, dy)/(speed + 1e-10)

def rk2_step(phi, dx, dy, Vx, Vy, dt, speed_fn):
    F1 = compute_rhs(phi, dx, dy, Vx, Vy, speed_fn)
    phi_star = phi + dt*F1
    F2 = compute_rhs(phi_star, dx, dy, Vx, Vy, speed_fn)
    return 0.5*phi + 0.5*(phi_star + dt*F2)

# Level set 
def run_levelset(phi0, X, Y, dx, dy,
                 Vx, Vy, T_end, save_times,
                 speed_fn, CFL=0.45):
    phi, t = phi0.copy(), 0.0
    snapshots = []
    save_times = sorted(save_times)
    save_index = 0

    if abs(save_times[0]) < 1e-12:
        snapshots.append((0.0, phi.copy()))
        save_index = 1

    while t < T_end - 1e-12:
        dt = get_dt(phi, dx, dy, Vx, Vy, speed_fn, CFL)

        if save_index < len(save_times):
            dt = min(dt, save_times[save_index]-t + 1e-12)
        phi = rk2_step(phi, dx, dy, Vx, Vy, dt, speed_fn)
        t += dt

        if save_index < len(save_times):
            if t >= save_times[save_index] - 1e-10:
                snapshots.append((float(t), phi.copy()))
                save_index += 1
    return snapshots

# MOL 
def mol_normal(x, y):
    dxs = np.roll(x,-1) - np.roll(x,1)
    dys = np.roll(y,-1) - np.roll(y,1)
    ds = np.sqrt(dxs**2 + dys**2) + 1e-12
    return dys/ds, -dxs/ds

def mol_rhs_mallet(x, y, Vx, Vy,
                   r0=0.165, cf=3.24,
                   n=1.5, alpha=0.5):
    nx, ny = mol_normal(x, y)
    Vm = np.sqrt(Vx**2 + Vy**2) + 1e-10
    cos_theta = np.clip((Vx*nx + Vy*ny)/Vm, -1, 1)
    theta = np.arccos(cos_theta)
    cp = np.maximum(cos_theta, 0)

    head = r0*(1 + cf*np.sqrt(Vm)*cp**n)
    flank = r0*(alpha + (1-alpha)*np.sin(theta))

    Un = np.maximum(
        np.where(np.abs(theta)<=np.pi/2, head, flank),
        1e-4
    )
    return Un*nx, Un*ny

def run_mol(x0, y0, r_init, Vx, Vy,
            T_end, N=120, dt=0.05,
            save_interval=6.0):
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = x0 + r_init*np.cos(theta)
    y = y0 + r_init*np.sin(theta)

    t = 0.0
    next_save = save_interval
    snapshots = [(0.0, x.copy(), y.copy())]

    while t < T_end - 1e-12:
        h = min(dt, T_end - t)
        k1x, k1y = mol_rhs_mallet(x, y, Vx, Vy)
        xs, ys = x + h*k1x, y + h*k1y
        k2x, k2y = mol_rhs_mallet(xs, ys, Vx, Vy)
        x = 0.5*x + 0.5*(xs + h*k2x)
        y = 0.5*y + 0.5*(ys + h*k2y)
        t += h

        if t >= next_save - 1e-10:
            snapshots.append((float(t), x.copy(), y.copy()))
            next_save += save_interval
    return snapshots

# Plots
def format_axes(ax, xlim=(-50,50), ylim=(0,100)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(alpha=0.15)

# Figure and 2
def fig1_and_fig2():
    X, Y, dx, dy = make_grid(100,100,30,30)
    Vx, Vy = 0.0, 3.0
    phi0 = phi_circle(X, Y, 0.0, 50.0, 10.0)
    snapshots_ls = run_levelset(
        phi0, X, Y, dx, dy,
        Vx, Vy, 30.0,
        [0,6,12,18,24,30],
        speed_mallet
    )
    snapshots_mol = run_mol(
        0.0, 50.0, 10.0,
        Vx, Vy, 30.0,
        N=120
    )

    # Figure 1
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,6))
    for _, phi in snapshots_ls:
        ax1.contour(X,Y,phi,levels=[0],colors=RED,linewidths=1.2)
    format_axes(ax1)
    ax1.set_title('Level-Set (Eulerian)')
    for _, xm, ym in snapshots_mol:
        ax2.plot(np.append(xm,xm[0]), np.append(ym,ym[0]),
                 color=RED, linewidth=1.0)
    format_axes(ax2)
    ax2.set_title('Lagrangian (MOL)')
    fig.suptitle('Figure 1. Wind-Driven Circular Fire Front Propagation',
                 fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.94])
    plt.savefig('figures/figure1.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2
    mol_dict = {round(t,1):(x,y) for t,x,y in snapshots_mol}
    fig, axes = plt.subplots(2,3,figsize=(10,8))
    for idx, (ts, phi) in enumerate(snapshots_ls[:6]):
        ax = axes.flatten()[idx]
        ax.contour(X,Y,phi,levels=[0],colors=RED,linewidths=1.2)
        key = min(mol_dict.keys(), key=lambda k: abs(k-round(ts,1)))
        xm, ym = mol_dict[key]
        ax.plot(np.append(xm,xm[0]), np.append(ym,ym[0]),
                color=BLUE, linestyle='--', linewidth=1.0)
        format_axes(ax)
        ax.set_title(f't = {ts:.0f} s')
    handles = [
        Line2D([0],[0],color=RED,linewidth=1.5,label='Level-Set'),
        Line2D([0],[0],color=BLUE,linestyle='--',
               linewidth=1.5,label='Lagrangian MOL')
    ]
    axes.flatten()[-1].legend(handles=handles,
                              loc='center',
                              frameon=False)
    fig.suptitle('Figure 2. Level-Set vs MOL Comparison',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/figure2.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 3
def fig3():
    X, Y, dx, dy = make_grid(100,160,100,160)
    Vx, Vy = 0.0, 1.5
    phi0 = combine(
        phi_line(X,Y,20.0),
        phi_circle(X,Y,0.0,40.0,7.0)
    )
    snapshots = run_levelset(
        phi0, X, Y, dx, dy,
        Vx, Vy, 80.0,
        [0,18,45,72],
        speed_australian
    )
    fig, axes = plt.subplots(2,2,figsize=(8,8))
    for ax, (ts, phi) in zip(axes.flatten(), snapshots):
        ax.contour(X,Y,phi,levels=[0],colors=RED,linewidths=1.2)
        format_axes(ax, ylim=(0,160))
        ax.set_title(f't = {ts:.0f} s')
    fig.suptitle('Figure 3. Merger of a Line Front and Spot Fire',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/figure3.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 4
def fig4():
    X, Y, dx, dy = make_grid(100,110,100,110)
    phi0 = combine(
        phi_circle(X,Y,-18.0,58.0,12.0),
        phi_circle(X,Y,18.0,58.0,12.0),
        phi_line(X,Y,30.0)
    )
    snapshots = run_levelset(
        phi0, X, Y, dx, dy,
        0.0, 0.0,
        110.0,
        [0,22,55,100],
        speed_australian
    )
    fig, axes = plt.subplots(2,2,figsize=(8,8))
    for ax, (ts, phi) in zip(axes.flatten(), snapshots):
        ax.contour(X,Y,phi,levels=[0],colors=RED,linewidths=1.2)
        format_axes(ax, ylim=(0,110))
        ax.set_title(f't = {ts:.0f} s')
    fig.suptitle('Figure 4. Formation of an Unburned Fuel Pocket',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/figure4.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 5
def fig5():
    X, Y, dx, dy = make_grid(100,100,160,160)
    phi0 = combine(
        phi_circle(X,Y,-20.0,5.0,4.0),
        phi_circle(X,Y,0.0,5.0,4.0),
        phi_circle(X,Y,20.0,5.0,4.0)
    )
    snapshots = run_levelset(
        phi0, X, Y, dx, dy,
        0.0, 3.0,
        60.0,
        list(range(0,61,5)),
        speed_australian
    )
    fig, ax = plt.subplots(figsize=(7,7))
    for _, phi in snapshots:
        ax.contour(X,Y,phi,levels=[0],colors=RED,linewidths=0.8)
    format_axes(ax)
    ax.annotate(
        '',
        xy=(-15,55),
        xytext=(-15,35),
        arrowprops=dict(arrowstyle='->',
                        color='steelblue',
                        lw=2.5)
    )
    fig.suptitle('Figure 5. Scalloped Fire Front Formation',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/figure5.png', dpi=300, bbox_inches='tight')
    plt.close()

# Convergence 
def convergence():
    grids = [20,30,40,60,80,100]
    heads = []
    for N in grids:
        X, Y, dx, dy = make_grid(100,100,N,N)
        phi0 = phi_circle(X,Y,0.0,50.0,10.0)
        snapshots = run_levelset(
            phi0, X, Y, dx, dy,
            0.0, 3.0,
            30.0,
            [30.0],
            speed_mallet
        )
        _, phi = snapshots[-1]
        mid = N//2
        col = phi[mid]
        ya = Y[mid]
        crossings = np.where(np.diff(np.sign(col)))[0]
        yh = np.nan

        if len(crossings):
            j = crossings[-1]
            yh = ya[j] + (ya[j+1]-ya[j]) * (-col[j]) / (
                col[j+1]-col[j] + 1e-12
            )
        heads.append(yh)

        print(f"N = {N:3d} --> y_head = {yh:.3f} m")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(grids, heads,
            marker='o',
            linewidth=1.8,
            markersize=5,
            color=RED,
            label='Computed')

    ax.axhline(93.9,
               color='black',
               linestyle='--',
               linewidth=1.2,
               label='Paper Reference')

    ax.set_xlabel('Grid Resolution')
    ax.set_ylabel('Fire Head Position (m)')
    ax.set_title('Grid Convergence Study')
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('figures/convergence.png',
                dpi=300,
                bbox_inches='tight')
    plt.close()

# Main
if __name__ == '__main__':

    print("\nRunning all simulations...\n")

    fig1_and_fig2()
    fig3()
    fig4()
    fig5()
    convergence()

    print("\nAll figures generated successfully.")
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
from matplotlib.animation import FuncAnimation

# Mesh
H = 1
L = 1
t = 0
alpha = 1
sum_upper_limit = 70

x = np.arange(0,L,0.01)
y = np.arange(0,H,0.01)
X, Y = np.meshgrid(x,y)

# Plot structure 
fig = plt.figure()
U = plt.axes(122, projection = '3d')
C = plt.axes(121)
C.set_aspect(aspect = 'equal' ,adjustable='box')

#BC
def a(y,t):
    return 0
def b(y,t): 
    return 0

#IC
def f(x,y): 
    return 1

def C_nm(n, m, L, H):
    n_1 = 70  

    pi = np.pi
    dx = L / (n_1 - 1)
    dy = H / (n_1 - 1)

    # Evaluate the integrand over the entire grid
    term = f(X, Y) - (a(Y, 0) + X / L * (b(Y, 0) - a(Y, 0)))
    integrand = term * np.sin(pi * n * X / L) * np.sin(pi * m * Y / H)
    integral = np.sum(integrand) * dx * dy
    return integral * (4 / (L * H))


#Solution
def u(x, y, t, sum_upper_lim, alpha, L, H):
    s1 = np.zeros_like(x)
    pi = np.pi
    for n in range(1, sum_upper_lim):
        for m in range(1, sum_upper_lim):
            s1 += C_nm(n, m, L, H) * np.sin(pi*n*x/L) * np.sin(pi*m*y/H) * np.exp(-alpha*((pi*n/L)**2 + (pi*m/H)**2)*t)
            # s1 += C_nm(x, y, n, m, L, H) * np.sin(pi*n*x/L) * np.sin(pi*m*y/H) * np.exp(-alpha*((pi*n/L)**2 + (pi*m/H)**2)*t)
    return s1 + a(y,t) + x/L*(b(y,t) - a(y,t))


#Plot
for axis in [U.xaxis.pane, U.yaxis.pane, U.zaxis.pane]:
    axis.fill = False
    axis.set_edgecolor('w')
U.grid(False)

U.set_xlabel('X', color = '#D81159')
U.set_ylabel('Y', color = '#D81159')
U.set_zlabel('u(x,y)', color = '#D81159')
C.set_xlabel('X', color = '#D81159')
C.set_ylabel('Y', color = '#D81159')

fig.suptitle('The Heat Equation with 2 Temporal Boundary Conditions', fontsize=24,
              bbox={'facecolor': '#D81159', 'alpha': 0.3, 'pad': 10})

Z = u(X, Y, 0, sum_upper_limit, alpha, L, H)  # Initial condition
surface = U.plot_surface(X, Y, Z, cmap='viridis', linewidth=1)
heatmap = C.pcolor(X, Y, Z, cmap='inferno')

# Colorbar
fig.colorbar(heatmap, ax=C, shrink=0.8, aspect=20)

def update(frame):
    t = frame / 150 
    Z = u(X, Y, t, sum_upper_limit, alpha, L, H)
    
    # Update surface plot
    U.clear()  
    U.set_zlim(-0.5, 1.0)
    U.set_xlabel('X', color='#D81159')
    U.set_ylabel('Y', color='#D81159')
    U.set_zlabel('u(x,y)', color='#D81159')
    U.plot_surface(X, Y, Z, cmap='viridis', linewidth=1)

    for axis in [U.xaxis.pane, U.yaxis.pane, U.zaxis.pane]:
        axis.fill = False
        axis.set_edgecolor('w')
    U.grid(False)
    
    # Update heatmap
    C.clear()  
    C.set_xlabel('X', color='#D81159')
    C.set_ylabel('Y', color='#D81159')
    C.pcolor(X, Y, Z, cmap='inferno')

# Create animation
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

anim = FuncAnimation(fig, update, frames=70, interval=200)

# Save as video (MP4)
anim.save("heat_equation_animation.mp4", writer=writer)

plt.show()

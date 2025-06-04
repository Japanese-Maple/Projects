'''numerical approximation of the two dimentional Heat Equation by Alex'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye, kron
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import cmasher as cmr

D3 = True

n=100
H = 2
alpha = 1
time = 1000
dx = H/(n+1)
dt = dx**2/(4*alpha)

x = np.linspace(0,H,n+2)
y = np.linspace(0,H,n+2)
X,Y = np.meshgrid(x,y)

def L(n):
    'Discrete 2D Laplacian Operator'

    I = eye(n)
    D = 4*eye(n) - eye(n, k=1) - eye(n, k=-1)
    L = kron(eye(n), D) - kron(eye(n, k=1), I) - kron(eye(n, k=-1), I)
    return L

def f(x,y):
    'Initial Condition'
    # return np.cos(x*pi+pi/2)*np.sin(y*pi)
    # return 1+0*x
    # return np.exp((x+y**2)*np.sin(30*x)*np.cos(30*y))*x
    # return np.sin(np.sqrt(3*x**2 + 3*y**2)) * np.cos(10*x) * np.sin(10*y)
    return (x-1)**2+(y-1)**2 + np.cos(x*y)


# Solution loop
v = np.zeros((time, n**2))
x_j = np.linspace(0+dx, H-dx, n)
x_v, y_v = np.meshgrid(x_j, x_j)
v1 = f(x_v, y_v).ravel()

v[0, :] = v1
for t in tqdm(range(1, time)):
    v[t, :] = -alpha*dt/(dx**2)*(L(n) @ v[t-1, :]) + v[t-1, :]

# Preparation for plotting
v = v.reshape(time, n, n)  
u_solution = np.pad(v, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

vmin = np.min(v1)
vmax = np.max(v1)

# Plot
if D3 == True:
    fig = plt.figure(figsize=(10, 10))
    U = fig.add_subplot(111, projection='3d')

else:
    fig, U = plt.subplots(figsize=(10,10))

def update(frame):
    U.clear()  
    
    # 3D view
    if D3 == True:
        U.view_init(elev=30, azim=frame * 0.3)
        U.set_zlim(vmin+0.1, vmax+0.1)
        U.set_xlabel('X', color='#D81159')
        U.set_ylabel('Y', color='#D81159')
        U.set_zlabel('u(x,y)', color="#FF005D")
        U.plot_surface(
                    X, Y, u_solution[frame, :, :],
                    cmap=cmr.guppy_r,
                    edgecolor='k',
                    linewidth=0.2,
                    vmin=vmin,
                    vmax=vmax,
                    antialiased=True
                    )

        for axis in [U.xaxis.pane, U.yaxis.pane, U.zaxis.pane]:
            axis.fill = False
            axis.set_edgecolor('w')
        U.grid(False)

    # 2D view
    else:
        U.set_aspect('equal')
        U.set_xlabel('X', color='#D81159')
        U.set_ylabel('Y', color='#D81159')
        U.contourf(X, Y, u_solution[frame, :], 
                cmap=cmr.guppy_r, 
                #    cmap = 'viridis',
                levels = 300,
                vmin=vmin, 
                vmax=vmax)
        # U.imshow(u_solution[frame, :], 
        #            cmap=cmr.guppy)

# Save
writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
anim = FuncAnimation(fig, update, frames=tqdm(range(time)), interval=200)
anim.save("HE2D_FDM.mp4", writer=writer)

plt.show()

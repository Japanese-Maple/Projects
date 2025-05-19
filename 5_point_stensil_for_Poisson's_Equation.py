import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye, kron, diags
from scipy.sparse import find
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D 
import sympy as sp
import inspect
import re

# -----------------------------------------------------------------------------------------------------------------------------------------
save_path = '5-point_stensil.pdf'
# -----------------------------------------------------------------------------------------------------------------------------------------

pi = np.pi

# Grid size

n = 10

l = 1
h = 1
dx = l/(n+1)
dy = h/(n+1)


# Function
def f(x,y):
    # return np.cos(x*pi+pi/2)*np.sin(y*pi)
    # return 1+0*x
    # return np.exp((x+y**2)*np.sin(30*x)*np.cos(30*y))*x
    return 2*x*y

def function_to_latex(func):
    source_lines = inspect.getsource(func).split('\n')
    expressions = []
    for line in source_lines:
        line = line.strip()
        if line.startswith('#'):
            continue
        match = re.match(r'return\s+(.*)', line)
        if match:
            expressions.append(match.group(1))
    latex_expressions = []
    for expr in expressions:
        expr = expr.replace('np.', '')  
        sympy_expr = sp.sympify(expr)
        latex_expr = sp.latex(sympy_expr)
        latex_expressions.append(latex_expr)
    return latex_expressions

f_latex = function_to_latex(f)[-1]  

# Matrix construction
T = 4 * eye(n) - diags([1, 1], [1, -1], shape=(n, n))
I = -eye(n)
A_sparse = kron(eye(n), T) + kron(eye(n, k=1), I) + kron(eye(n, k=-1), I)

# Solution to u_xx + u_yy = f(x,y)
x = np.linspace(0,l,n+2)
y = np.linspace(0,h,n+2)
X,Y = np.meshgrid(x,y)

b = []
for j in range(n):
    for i in range(n):
        b.append(f(dx*(i+1), dy*(j+1)))

u = spsolve(A_sparse, np.array(b) * dx**2)
Z_interior = u.reshape(n,n)
Z_u = np.zeros((n+2, n+2))
Z_u[1:-1, 1:-1] = Z_interior

# -----------------------------------------------------------------------------------------------------------------------------------------

fig = plt.figure(figsize=(15, 15))
gs = fig.add_gridspec(2, 2, width_ratios=[0.5, 0.5])

# Create subplots
U = [[fig.add_subplot(gs[i, j], projection='3d' if (i, j) == (1, 1) or (i, j) == (0, 1) else None) for j in range(2)] for i in range(2)]

plt.subplots_adjust(wspace=0.1, hspace=0.3)


U[0][1].grid(False)
U[0][1].set_xlabel('X', color='#D81159')
U[0][1].set_ylabel('Y', color='#D81159')
U[0][1].set_zlabel('$u(x,y)$', color='#D81159')
U[0][1].plot_surface(X, Y, Z_u, cmap='viridis', linewidth=1)
U[0][1].set_title(fr"$\frac{{\partial^2 u}}{{\partial x^2}} + \frac{{\partial^2 u}}{{\partial y^2}} = {f_latex}$", fontsize=20)

for axis in [U[0][1].xaxis.pane, U[0][1].yaxis.pane, U[0][1].zaxis.pane]:
    axis.fill = False
    axis.set_edgecolor('w')

rows, cols, values = find(A_sparse)
U[0][0].scatter(rows, cols, c=values, marker = 's', s=300 * U[0][0].get_position().width / n, cmap='viridis')
U[0][0].set_aspect('equal')
# U[0][0].spy(A_sparse, markersize=300 * U[0][0].get_position().width / n)
U[0][0].set_title(f"2D Poisson Equation (n={n})")


U[1][0].set_title('Heatmap')
U[1][0].imshow(Z_u, cmap='viridis', interpolation='nearest')
U[1][0].set_xticks([0,n])
U[1][0].set_xlabel('X')
U[1][0].set_yticks([0,n])
U[1][0].set_ylabel('Y')

U[1][1].grid(False)
U[1][1].set_title(fr"$f(x,y) = {f_latex}$")
U[1][1].plot_surface(X, Y, f(X,Y), cmap='inferno', linewidth=1)
U[1][1].set_xlabel('X', color='#D81159')
U[1][1].set_ylabel('Y', color='#D81159')
U[1][1].set_zlabel('$f(x,y)$', color='#D81159')

for axis in [U[1][1].xaxis.pane, U[1][1].yaxis.pane, U[1][1].zaxis.pane]:
    axis.fill = False
    axis.set_edgecolor('w')

fig.suptitle("5-point stencil for Poisson's Equation", fontsize = 23)
fig.savefig(f'{save_path}')
plt.show()


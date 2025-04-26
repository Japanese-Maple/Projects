import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye, kron, diags
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D 
import sympy as sp
import inspect
import re

pi = np.pi

# Grid size

n = 1000

l = 1
h = 1
dx = l/(n+1)
dy = h/(n+1)


# Function
def f(x,y):
    # return np.cos(x*pi+pi/2)*np.sin(y*pi)
    # return 1+0*x
    return np.exp((x+y**2)*np.sin(30*x)*np.cos(30*y))*x

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
        expr = expr.replace('np.', '')  # Remove 'np.' prefix
        sympy_expr = sp.sympify(expr)
        latex_expr = sp.latex(sympy_expr)
        latex_expressions.append(latex_expr)
    return latex_expressions

f_latex = function_to_latex(f)[-1]  # Last (active) expression

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

# Plot

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])  # Fixed ratio
U = [fig.add_subplot(gs[0, i], projection='3d' if i==2 else None) for i in range(3)]

plt.subplots_adjust(wspace=0.4)  

for axis in [U[2].xaxis.pane, U[2].yaxis.pane, U[2].zaxis.pane]:
    axis.fill = False
    axis.set_edgecolor('w')
U[2].grid(False)

U[2].set_xlabel('X', color = '#D81159')
U[2].set_ylabel('Y', color = '#D81159')
U[2].set_zlabel('u(x,y)', color = '#D81159')

U[2].plot_surface(X, Y, Z_u, cmap = 'viridis', linewidth = 1)
# U[2].plot_surface(X, Y, f(X,Y), cmap = 'inferno', linewidth = 1)
U[2].set_title(fr"$\frac{{\partial^2 u}}{{\partial x^2}} + \frac{{\partial^2 u}}{{\partial y^2}} = {f_latex}$", 
               fontsize=20)

U[0].spy(A_sparse, markersize= 300*U[0].get_position().width/(n))
U[0].set_title(f"2D Poisson Equation (n={n})")
U[1].imshow(Z_u, cmap='viridis', interpolation='nearest') 
for i in range(1):
    U[i].grid(False)
fig.suptitle("5-point stensil for Poisson's Equation")
plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import sympy as sp
import inspect
import re

#-----------------------------------------------------------------------------
#DEFINE THE PARAMETERS OF A PLANE
H = 1
L = 1

#-----------------------------------------------------------------------------
fig = plt.figure()
U = plt.axes(122, projection = '3d')
C = plt.axes(121)
C.set_aspect(aspect = 'equal' ,adjustable='box')

#-----------------------------------------------------------------------------
# DEFINE LAMBDAS/FREQUENCIES
μ = np.pi/H
o = np.pi/L

#-----------------------------------------------------------------------------
x = np.arange(0,L,0.01)
y = np.arange(0,H,0.01)
X, Y = np.meshgrid(x,y)

#-----------------------------------------------------------------------------
#FUNCTIONS FOR BOUNDARY CONDITIONS
def f1(x):
    # return ((2*x-1)**2+1)/2
    # return -2*(((2*x-1)**4)-((2*x-1)**2))
    # return np.sin(1/(x-0.5))
    # return 0.5 * np.sqrt(1-(2*x-1)**2)
    #return np.exp(-4*((x-1/2)**2))
    # return np.exp(-4*((x-1/2)**2))
    return np.sin(np.pi*3*x)
def f2(x): 
    return np.sin(np.pi*3*x)
def g1(x):
    return np.sin(np.pi*3*x)
def g2(x):
    return np.sin(np.pi*3*x)
iterations = 200
#-----------------------------------------------------------------------------
#FUNCTION EXTRACTION 
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

f1_latex = function_to_latex(f1)[-1]  # Last (active) expression
f2_latex = function_to_latex(f2)[-1]
g1_latex = function_to_latex(g1)[-1]
g2_latex = function_to_latex(g2)[-1]

#-----------------------------------------------------------------------------
#COEFFICIENTS
def A_n(f, n, w, LH):
    """
    Parameters
    ----------
    f : function 
    n : natural number that goes into sin() function
    w : the scalar inside of sin() function
    LH : either H or L
    """
    n_1 = 10000
    dx = LH / (n_1 - 1)
    s = 0
    for m in range(n_1):
        t = f(dx * m)*np.sin(w*n*(dx * m)) * dx
        s += t
    return s

#-----------------------------------------------------------------------------
#SOLUTION
def u(f1, f2, g1, g2, n, x, y):
    s1 = np.zeros_like(x)
    s2 = np.zeros_like(y)
    for i in range(1, n):
        s1 += ((2*np.sin(o*x*i))/(L*np.sinh(o*H*i)))*(np.sinh(o*i*(H-y))*A_n(f1, i, o, L) + np.sinh(o*y*i)*A_n(f2, i, o, L))
    
        s2 += ((2*np.sin(μ*y*i))/(H*np.sinh(μ*L*i)))*(np.sinh(μ*i*(L-x))*A_n(g1, i, μ, H) + np.sinh(μ*i*x)*A_n(g2, i, μ, H))
    return s1+s2

#-----------------------------------------------------------------------------
U.xaxis.pane.fill = False
U.yaxis.pane.fill = False
U.xaxis.pane.set_edgecolor('w')
U.yaxis.pane.set_edgecolor('w')
U.zaxis.pane.set_color('w')
U.grid(False)

#-----------------------------------------------------------------------------
U.set_xlabel('X', color = '#D81159')
U.set_ylabel('Y', color = '#D81159')
U.set_zlabel('u(x,y)', color = '#D81159')
C.set_xlabel('X', color = '#D81159')
C.set_ylabel('Y', color = '#D81159')

#-----------------------------------------------------------------------------
Z = u(f1,f2,g1,g2, iterations, X, Y)

u_x_y = U.plot_surface(X, Y, Z, cmap = 'viridis', linewidth = 1)
U.plot(x, f1(x), zs=0, zdir='y', color='b', linewidth = 2)
U.plot(x, f2(x), zs=H, zdir='y', color='g', linewidth = 2)
U.plot(y, g1(y), zs=0, zdir='x', color='r', linewidth = 2)
U.plot(y, g2(y), zs=L, zdir='x', color='k', linewidth = 2)

color = C.pcolor(X, Y, Z, cmap = 'inferno', label ='u(x,y)')

fig.colorbar(color, ax=C, shrink=0.73, aspect=20)

fig.suptitle('Laplace Equation with 4 Boundary Conditions', fontsize=24,
              bbox={'facecolor': '#D81159', 'alpha': 0.3, 'pad': 10})
U.legend(['u(x,y)',
          f'$f_{1}={f1_latex}$', 
          f'$f_{2}={f2_latex}$', 
          f'$g_{1}={g1_latex}$', 
          f'$g_{2}={g2_latex}$'])

plt.show()

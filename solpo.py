import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye, kron
from scipy.sparse import find
import cmasher as cmr
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D 
from scipy.sparse.linalg import spsolve
import sympy as sp
import inspect
import re

class Poisson_Equation():
    def __init__(self, H, n, IC):
        self.H = H
        self.n = n
        self.IC = IC
        self.start_time = 0

    @staticmethod
    def _function_to_latex(func):
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
    
    def L_h2(self):
        """Discrete 2D Laplacian Operator: Order O(h²)
        
        Parameters
        ----------
        n : int
            Number of spatial subdivisions along each axis (creates an n² × n² grid).

        Returns
        -------
        np.ndarray
            Laplacian of shape (n², n²)"""
        
        I = eye(self.n)
        D = 4*eye(self.n) - eye(self.n, k=1) - eye(self.n, k=-1)
        L = kron(eye(self.n), D) - kron(eye(self.n, k=1), I) - kron(eye(self.n, k=-1), I)
        return L
    
    def solve_Poisson_Equation(self):
        dx = self.H / (self.n + 1)
        L = self.L_h2()

        x = np.linspace(0 + dx, self.H - dx, self.n)
        y = np.linspace(0 + dx, self.H - dx, self.n)
        X,Y = np.meshgrid(x, y)

        u = spsolve(L, self.IC(X,Y).ravel() * dx**2)
        Z_interior = u.reshape(self.n, self.n)
        Z_u = np.zeros((self.n + 2, self.n + 2))
        Z_u[1:-1, 1:-1] = Z_interior

        return Z_u
    
    def render_Poisson_Eqn(self, solution, save_path):
        L = self.L_h2()
        function = Poisson_Equation._function_to_latex(self.IC)[0] 
        
        x = np.linspace(0, self.H, self.n + 2)
        y = np.linspace(0, self.H, self.n + 2)
        X,Y = np.meshgrid(x,y)

        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(2, 2, width_ratios=[0.5, 0.5])

        # Subplots
        U = [[fig.add_subplot(gs[i, j], projection='3d' if (i, j) == (1, 1) or (i, j) == (0, 1) else None) for j in range(2)] for i in range(2)]

        plt.subplots_adjust(wspace=0.1, hspace=0.3)


        U[0][1].grid(False)
        U[0][1].set_xlabel('X', color='#D81159')
        U[0][1].set_ylabel('Y', color='#D81159')
        U[0][1].set_zlabel('$u(x,y)$', color='#D81159')
        U[0][1].plot_surface(X, Y, solution, cmap='viridis', linewidth=1)
        U[0][1].set_title(fr"$\frac{{\partial^2 u}}{{\partial x^2}} + \frac{{\partial^2 u}}{{\partial y^2}} = {function}$", fontsize=20)

        for axis in [U[0][1].xaxis.pane, U[0][1].yaxis.pane, U[0][1].zaxis.pane]:
            axis.fill = False
            axis.set_edgecolor('w')

        rows, cols, values = find(L)
        U[0][0].scatter(rows, cols, c=values, marker = 's', s=300 * U[0][0].get_position().width / n, cmap='viridis')
        U[0][0].set_aspect('equal')
        U[0][0].invert_yaxis()
        # U[0][0].spy(A_sparse, markersize=300 * U[0][0].get_position().width / n)
        U[0][0].set_title(f"2D Poisson Equation (n={n})")


        U[1][0].set_title('Heatmap')
        U[1][0].imshow(solution, cmap='viridis', interpolation='nearest')
        U[1][0].set_xticks([0,n])
        U[1][0].set_xlabel('X')
        U[1][0].set_yticks([0,n])
        U[1][0].set_ylabel('Y')

        U[1][1].grid(False)
        U[1][1].set_title(fr"$f(x,y) = {function}$")
        U[1][1].plot_surface(X, Y, self.IC(X,Y), cmap='inferno', linewidth=1)
        U[1][1].set_xlabel('X', color='#D81159')
        U[1][1].set_ylabel('Y', color='#D81159')
        U[1][1].set_zlabel('$f(x,y)$', color='#D81159')

        for axis in [U[1][1].xaxis.pane, U[1][1].yaxis.pane, U[1][1].zaxis.pane]:
            axis.fill = False
            axis.set_edgecolor('w')

        fig.suptitle("5-point stencil for Poisson's Equation", fontsize = 23)
        fig.savefig(f'{save_path}')


if __name__ == "__main__":
    # Example usage

    import random

    n = 90
    H = 2
    save_path = '5-point_stensil.pdf'

    # def random_math_function(x, y):

    #     # Mathematical operations and functions to mix
    #     funcs = [
    #         lambda z: np.sin(z),
    #         lambda z: np.cos(z),
    #         lambda z: np.tan(z % (np.pi / 2 - 0.1)), 
    #         lambda z: np.exp(-z**2),                  
    #         lambda z: np.log(np.abs(z) + 1),          
    #         lambda z: z**2,
    #         lambda z: np.sqrt(np.abs(z))
    #     ]

    #     # Randomly select and apply functions
    #     f1 = random.choice(funcs)
    #     f2 = random.choice(funcs)

    #     return f1(x) + f2(y)

    # def f(x, y):
    #     # Calculate squared distances from the centers
    #     r  = np.random.uniform(0.05, 0.3)
    #     d  = np.random.uniform(0.3, 1)
    #     d1 = np.random.uniform(1, 1.7)

    #     dist_pos = (x - d)**2 + (y - d)**2
    #     dist_neg = (x - d1)**2 + (y - d)**2

    #     # Create regions where distance squared is less than radius squared
    #     result = np.where(dist_pos <= r**2, 1, 0)
    #     result = np.where(dist_neg <= r**2, -1, result)

    #     return result

    def f(x, y):    
        return (np.sin(x + y) - 3 * np.cos(15 * x)) * 2*np.exp(-3*((x-H/2)**2+(y-H/2)**2))
    
    PE = Poisson_Equation(H=H, n=n, IC=f)
    sol = PE.solve_Poisson_Equation()
    PE.render_Poisson_Eqn(sol, save_path)
    

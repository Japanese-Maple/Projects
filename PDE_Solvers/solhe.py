import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye, kron
from tqdm import tqdm
import cmasher as cmr
import os
import matplotlib
import shutil
matplotlib.use('Agg')
from joblib import Parallel, delayed
from PIL import Image
import time

class Heat_Equation():
    def __init__(self, H, n, alpha, simulation_time, IC, upper, lower, right, left, dpi: int = 100, number_of_workers: int = 15, **mask_kwargs):
        self.H = H
        self.n = n
        self.alpha = alpha
        self.simulation_time = simulation_time
        self.IC = IC
        self.upper = upper
        self.lower = lower
        self.right = right
        self.left = left
        self.dpi = dpi
        self.num_workers = number_of_workers
        self.mask_kwargs = mask_kwargs
        self.start_time = 0

    def _get_runtime(self):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(elapsed, 60)
        hrs, mins = divmod(mins, 60)
        print(f"Total execution time: {int(hrs)}h {int(mins)}m {int(secs)}s")

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
    
    def solve_Heat_Equation_4BC(self):

        self.start_time = time.time()

        dx = self.H / (self.n + 1)
        dt = dx**2 / (4*self.alpha)

        v = np.zeros((self.simulation_time, self.n**2))
        x_j_outer_upper = np.linspace(0, self.H, self.n)
        x_j_outer_right = np.linspace(0 + dx, self.H - dx, self.n - 2)
        x_j_inner = np.linspace(0 + dx, self.H - dx, self.n - 2)
        x_v, y_v = np.meshgrid(x_j_inner, x_j_inner)
        inner_domain = self.IC(x_v, y_v)
        L = self.L_h2()

        # v1 = np.vstack([self.lower(x_j_outer_upper, 0), 
        #                 np.hstack([self.left(x_j_inner, 0), 
        #                            inner_domain, 
        #                            self.right(x_j_inner, 0)]), 
        #                 self.upper(x_j_outer_upper, 0)])        # N x N 

        left_bc = self.left(x_j_inner, 0).reshape(-1, 1)
        right_bc = self.right(x_j_inner, 0).reshape(-1, 1)
        top_bc = self.upper(x_j_outer_upper, 0).reshape(1, -1)
        bottom_bc = self.lower(x_j_outer_upper, 0).reshape(1, -1)

        middle = np.hstack([left_bc, inner_domain, right_bc])
        v1 = np.vstack([bottom_bc, middle, top_bc])

        v[0, :] = v1.ravel() # N x N --> <1, N**2>
        for t in tqdm(range(1, self.simulation_time)):
            v[t, :] = -self.alpha*dt/(dx**2)*(L @ v[t-1, :]) + v[t-1, :]

            u_t = v[t, :].reshape(self.n, self.n) 
            u_t[-1, :]      = self.upper(x_j_outer_upper, t * dt)          # top row
            u_t[0, :]       = self.lower(x_j_outer_upper, t * dt)          # bottom row
            u_t[1:-1, 0]    = self.left(x_j_outer_right, t * dt).ravel()   # left column
            u_t[1:-1, -1]   = self.right(x_j_outer_right, t * dt).ravel()  # right column

            v[t, :] = u_t.ravel()

        u_solution = v.reshape(self.simulation_time, self.n, self.n)
        vmin = np.min(v1)
        vmax = np.max(v1)

        self._get_runtime()
        return u_solution, vmax, vmin

    def render_Heat_Eqn_2D(self, solution: np.ndarray, vmax: float, vmin: float):

        self.start_time = time.time()

        x = np.linspace(0, self.H, self.n)
        y = np.linspace(0, self.H, self.n)
        X,Y = np.meshgrid(x,y)

        output_dir = "frames_HE_2D"
        os.makedirs(output_dir, exist_ok=True)

        def render_frame_2D(frame):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.contourf(X, Y, solution[frame, :, :], cmap=cmr.guppy_r, levels=300, vmin=vmin, vmax=vmax) #cmr.guppy_r
            plt.savefig(f"{output_dir}/frame_{frame:04d}.png", bbox_inches='tight', pad_inches=0, dpi=self.dpi)
            plt.close(fig)   

        print("Rendering frames in parallel...")
        Parallel(n_jobs=self.num_workers)(delayed(render_frame_2D)(t) for t in tqdm(range(self.simulation_time), desc="Rendering"))

        sample_frame = Image.open(f"{output_dir}/frame_0000.png")
        width, height = sample_frame.size

        # Ensure both dimensions are divisible by 2 (even numbers)
        even_width = width if width % 2 == 0 else width + 1
        even_height = height if height % 2 == 0 else height + 1

        print(f"Sample frame resolution: {width}x{height} → using {even_width}x{even_height} for video")

        video_filename = 'HE_2D_2D.mp4'
        print("Stitching frames into video...")

        ffmpeg_cmd = f"""
        ffmpeg -loglevel error -y -framerate 30 -i {output_dir}/frame_%04d.png \
        -vf "scale={even_width}:{even_height}:flags=lanczos" \
        -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p {video_filename}
        """

        os.system(ffmpeg_cmd)

        # Clean up
        print("Deleting frames folder...")
        shutil.rmtree(output_dir)

        # Saved to
        print(f"Video saved to {video_filename}")
        self._get_runtime()

    def render_Heat_Eqn_3D(self, solution: np.ndarray, vmax: float, vmin: float):

        self.start_time = time.time()

        x = np.linspace(0, self.H, self.n)
        y = np.linspace(0, self.H, self.n)
        X,Y = np.meshgrid(x,y)

        output_dir = "frames_HE_3D"
        os.makedirs(output_dir, exist_ok=True)
        
        def render_frame_3D(frame):
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=30, azim=frame * 0.3)
            ax.set_zlim(vmin + 0.1, vmax + 0.1)
            ax.set_xlabel('X', color='#D81159')
            ax.set_ylabel('Y', color='#D81159')
            ax.set_zlabel('u(x,y)', color="#FF005D")
            ax.plot_surface(X, Y, solution[frame, :, :], cmap=cmr.guppy_r,
                            edgecolor='k', linewidth=0.2, vmin=vmin, vmax=vmax,
                            antialiased=True)
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False
                pane.set_edgecolor('w')
            ax.grid(False)
            plt.savefig(f"{output_dir}/frame_{frame:04d}.png", bbox_inches='tight', pad_inches=0, dpi=self.dpi)
            plt.close(fig)

        print("Rendering frames in parallel...")
        Parallel(n_jobs=self.num_workers)(delayed(render_frame_3D)(t) for t in tqdm(range(self.simulation_time), desc="Rendering"))

        sample_frame = Image.open(f"{output_dir}/frame_0000.png")
        width, height = sample_frame.size

        # Ensure both dimensions are divisible by 2 (even numbers)
        even_width = width if width % 2 == 0 else width + 1
        even_height = height if height % 2 == 0 else height + 1

        print(f"Sample frame resolution: {width}x{height} → using {even_width}x{even_height} for video")

        video_filename = 'HE_2D_3D.mp4'
        print("Stitching frames into video...")

        ffmpeg_cmd = f"""
        ffmpeg -loglevel error -y -framerate 30 -i {output_dir}/frame_%04d.png \
        -vf "scale={even_width}:{even_height}:flags=lanczos" \
        -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p {video_filename}
        """

        os.system(ffmpeg_cmd)

        # Clean up
        print("Deleting frames folder...")
        shutil.rmtree(output_dir)

        # Saved to
        print(f"Video saved to {video_filename}")
        self._get_runtime()


if __name__ == "__main__":
    # Example usage
    N = 90
    H = 2
    alpha = 1
    time_s = 1500
    
    def f(x,y):
        'Initial Condition'
        # return np.where((x > 0.1) & (x < 1) & (y > 1) & (y < 1.9), 3, 0)
        # return np.ones_like(x)
        return np.zeros_like(x)

    def a(x,t):
        'Upper Boundary Condition'
        # return -x*(x-H)*4*np.exp(-5*t)
        return 13*np.sin(2*np.pi*x)*np.cos(50*t)

    def b(y,t):
        'Right Boundary Condition'
        # function = -y*(y-H)*4*np.exp(-3*t)
        function = 13*np.sin(2*np.pi*y)*np.cos(50*t)
        return np.reshape(function, (N-2, 1))

    def c(x,t):
        'Lower Boundary Condition'
        # return np.zeros_like(x)
        # return 5*np.ones_like(x)
        return 13*np.sin(2*np.pi*x)*np.cos(50*t)

    def d(y,t):
        'Left Boundary Condition'
        # function = np.zeros_like(y)
        function = 13*np.sin(2*np.pi*y)*np.cos(50*t)
        return np.reshape(function, (N-2, 1))
    
    HE = Heat_Equation(H=H, n=N, alpha=alpha, simulation_time=time_s, IC=f, upper=a, lower=c, right=b, left=c, dpi=100)
    sol, vmax, vmin = HE.solve_Heat_Equation_4BC()
    # HE.render_Heat_Eqn_3D(sol, vmax, vmin)
    HE.render_Heat_Eqn_2D(sol, vmax, vmin)

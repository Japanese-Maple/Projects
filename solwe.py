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

class Wave_Equation():
    def __init__(self, H, n, simulation_time, IC, mask, dpi: int = 100, number_of_workers: int = 15, **mask_kwargs):
        self.H = H
        self.n = n
        self.simulation_time = simulation_time
        self.IC = IC
        self.dpi = dpi
        self.num_workers = number_of_workers
        self.mask = mask
        self.mask_kwargs = mask_kwargs
        self.start_time = 0

    def _get_runtime(self):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(elapsed, 60)
        hrs, mins = divmod(mins, 60)
        print(f"Total execution time: {int(hrs)}h {int(mins)}m {int(secs)}s")

    def _get_mask(self, x, y):
        return self.mask(x, y, **self.mask_kwargs)
    
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

    def solve_Wave_Equation_mask_DBC(self):
        """
        Solves the 2D wave equation with Dirichlet boundary conditions over a masked spatial domain.

        Parameters
        ----------
        time_of_simulation : int
            Total number of time steps to simulate.
        n : int
            Number of spatial subdivisions along each axis (creates an n × n grid).
        H : float
            Length of the square domain's edge.
        mask : function
            A function that defines the valid spatial domain (e.g., a circular mask).
        f : function
            A function that sets the initial state of the wave.
        L : function
            A function that creates the discrete Lagrangian operator (of size n² × n²).

        Returns
        -------
        np.ndarray
            Array of shape (time_of_simulation, n, n) containing the wave values at each time step.
        float
            Maximum value (vmax) for consistent color scaling in visualizations.
        float
            Minimum value (vmin) for consistent color scaling in visualizations.
        """
        self.start_time = time.time()

        dx = self.H/(self.n+1)
        dt = dx/4
        L = self.L_h2()

        # grid
        v = np.zeros((self.simulation_time, self.n**2))
        x_j = np.linspace(0 + dx, self.H - dx, self.n)
        x_v, y_v = np.meshgrid(x_j, x_j)

        # mask
        domain_mask = self._get_mask(x_v, y_v)
        
        # imposing IC
        v0 = self.IC(x_v, y_v).ravel()
        v[0, :] = v0

        v[1, :] = v0 - dt**2/(2 * dx**2) * (L @ v0)
        for t in tqdm(range(2, self.simulation_time)):
            v[t, :] = 2*v[t-1, :] - (dt**2)/(dx**2)*(L @ v[t-1, :]) - v[t-2, :]
            v[t, :] = np.where(domain_mask == 1, v[t, :].reshape(self.n,self.n), 0).ravel()

        # Preparation for plotting
        v = v.reshape(self.simulation_time, self.n, self.n)  
        u_solution = np.pad(v, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
        vmin =  np.min(v0)
        vmax =  np.max(v0)

        self._get_runtime()
        return u_solution, vmax, vmin

    def render_Wave_Eqn_2D(self, solution: np.ndarray, vmax: float, vmin: float):

        self.start_time = time.time()

        x = np.linspace(0, self.H, self.n+2)
        y = np.linspace(0, self.H, self.n+2)
        X,Y = np.meshgrid(x,y)

        output_dir = "frames_WE_2D"
        os.makedirs(output_dir, exist_ok=True)

        def render_frame_2D(frame):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.contourf(X, Y, solution[frame, :, :], cmap=cmr.redshift, levels=300, vmin=vmin, vmax=vmax) #cmr.redshift
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

        video_filename = 'WE2D_FDM_2D_circular.mp4'
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

    def render_Wave_Eqn_3D(self, solution: np.ndarray, vmax: float, vmin: float):

        self.start_time = time.time()

        x = np.linspace(0, self.H, self.n+2)
        y = np.linspace(0, self.H, self.n+2)
        X,Y = np.meshgrid(x,y)

        output_dir = "frames_WE_3D"
        os.makedirs(output_dir, exist_ok=True)
        
        def render_frame_3D(frame):
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=30, azim=frame * 0.3)
            ax.set_zlim(vmin + 0.1, vmax + 0.1)
            ax.set_xlabel('X', color='#D81159')
            ax.set_ylabel('Y', color='#D81159')
            ax.set_zlabel('u(x,y)', color="#FF005D")
            ax.plot_surface(X, Y, solution[frame, :, :], cmap=cmr.redshift,
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

        video_filename = 'WE2D_FDM_3D_circular.mp4'
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

    N = 100
    T = 1000
    H = 2

    def mask(x, y):
            x_c, y_c, R = H/2, H/2, H/2
            return np.where(0.5*(x - x_c)**2 + (y - y_c)**2 <= R**2 / 2, 1, 0)

    def IC(x, y):
        a = 1.73205080757
        func = 2*np.exp(-50*((x - a)**2 + (y - H/2)**2)) + 2*np.exp(-50*((x - (2 - a))**2 + (y - H/2)**2))
        mask1 = mask(x, y)
        return np.where(mask1 == 1, func, 0)


    WE = Wave_Equation(H=H, n=N, simulation_time=T, IC=IC, mask=mask, number_of_workers=15)
    sol, m, h = WE.solve_Wave_Equation_mask_DBC()
    WE.render_Wave_Eqn_2D(sol, m, -m)
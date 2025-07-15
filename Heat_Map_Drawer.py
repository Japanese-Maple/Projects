import tkinter as tk
import numpy as np
import cmasher as cmr

class HeatMapDrawer:
    def __init__(self, n=28, canvas_size=560):
        self.n = n
        self.canvas_size = canvas_size  # Fixed canvas size
        self.cell_size = self.canvas_size / self.n  # Float for precision
        self.offset_x = 0  # Optional: center grid by computing unused space
        self.offset_y = 0

        self.heat = np.zeros((n, n), dtype=np.float32)
        self.colormap = cmr.guppy_r
        self.current_value = 1.0
        self.pen_radius = 3
        self.gain = 1.0

        self.root = tk.Tk()
        self.root.title("Interactive Heatmap Drawer")

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        self.setup_controls()
        self.draw_all()

    def setup_controls(self):
        tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL,
                 label="Heat Level", command=self.update_value).pack()
        tk.Scale(self.root, from_=1, to=10, orient=tk.HORIZONTAL,
                 label="Pen Radius", command=self.update_radius).pack()
        tk.Scale(self.root, from_=1, to=90, resolution=10,
                 orient=tk.HORIZONTAL, label="Brush Strength %", command=self.update_gain).pack()

        frame = tk.Frame(self.root)
        frame.pack(pady=5)
        tk.Button(frame, text="Export Heat Matrix", command=self.export).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Reset Canvas", command=self.reset).pack(side=tk.LEFT, padx=5)

    def update_value(self, val):
        self.current_value = float(val) / 100.0

    def update_radius(self, val):
        self.pen_radius = int(val)

    def update_gain(self, val):
        self.gain = float(val) / 100.0

    def apply_gaussian(self, cx, cy, radius=3, sigma=2.0):
        radius = radius * 2
        x = np.arange(-radius, radius)
        y = np.arange(-radius, radius)
        xv, yv = np.meshgrid(x, y)
        mask = np.sqrt(xv**2 + yv**2) <= radius

        kernel = np.exp(-(xv**2 + yv**2) / (2 * sigma**2))
        kernel *= mask
        kernel /= kernel.max()
        kernel *= self.gain

        for dy in range(-radius, radius):
            for dx in range(-radius, radius):
                i, j = cy + dy, cx + dx
                if 0 <= i < self.n and 0 <= j < self.n:
                    influence = kernel[dy + radius, dx + radius] * self.current_value
                    self.heat[i, j] = min(1.0, self.heat[i, j] + influence)
                    self.draw_cell(j, i, self.heat[i, j])

    def paint(self, event):
        x = int((event.x - self.offset_x) / self.cell_size)
        y = int((event.y - self.offset_y) / self.cell_size)
        if 0 <= x < self.n and 0 <= y < self.n:
            sigma = max(0.5, self.pen_radius / 1.2)
            self.apply_gaussian(x, y, radius=self.pen_radius, sigma=sigma)

    def map_heat_to_hex(self, heat_val):
        r, g, b = self.colormap(heat_val)[:3]
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

    def draw_cell(self, x, y, value):
        x0 = self.offset_x + x * self.cell_size
        y0 = self.offset_y + y * self.cell_size
        x1 = x0 + self.cell_size
        y1 = y0 + self.cell_size
        color = self.map_heat_to_hex(value)
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

    def draw_all(self):
        for y in range(self.n):
            for x in range(self.n):
                self.draw_cell(x, y, self.heat[y, x])

    def reset(self):
        self.heat.fill(0)
        self.canvas.delete("all")
        self.draw_all()

    def export(self):
        print(self.heat)
        return self.heat

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    drawer = HeatMapDrawer(n=100)
    drawer.run()
    print(drawer.heat)


class HeatMapDrawer_smooth:
    def __init__(self, n=28, canvas_size=560):
        self.n = n
        self.canvas_size = canvas_size
        self.cell_size = self.canvas_size // self.n
        self.heat = np.zeros((n, n), dtype=np.float32)
        self.colormap = cmr.guppy_r
        self.current_value = 1.0
        self.pen_radius = 3
        self.gain = 1.0  # Brush strength
        self.last_x = None
        self.last_y = None

        self.root = tk.Tk()
        self.root.title("Interactive Heatmap Drawer")

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.clear_last_point())

        self.setup_controls()
        self.draw_all()

    def setup_controls(self):
        level_slider = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL,
                                label="Heat Level", command=self.update_value)
        level_slider.set(100)
        level_slider.pack()

        radius_slider = tk.Scale(self.root, from_=1, to=30, orient=tk.HORIZONTAL,
                                 label="Pen Radius", command=self.update_radius)
        radius_slider.set(self.pen_radius)
        radius_slider.pack()

        gain_slider = tk.Scale(self.root, from_=1, to=90, resolution=10,
                               orient=tk.HORIZONTAL, label="Brush Strength %", command=self.update_gain)
        gain_slider.set(31)
        gain_slider.pack()

        frame = tk.Frame(self.root)
        frame.pack(pady=5)
        tk.Button(frame, text="Export Heat Matrix", command=self.export).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Reset Canvas", command=self.reset).pack(side=tk.LEFT, padx=5)

    def update_value(self, val):
        self.current_value = float(val) / 100.0

    def update_radius(self, val):
        self.pen_radius = int(val)

    def update_gain(self, val):
        self.gain = float(val) / 100.0

    def apply_gaussian(self, cx, cy, radius=3, sigma=2.0):
        radius = radius * 2
        x = np.arange(-radius, radius)
        y = np.arange(-radius, radius)
        xv, yv = np.meshgrid(x, y)
        distance = np.sqrt(xv**2 + yv**2)
        mask = distance <= radius

        kernel = np.exp(-(xv**2 + yv**2) / (2 * sigma**2))
        kernel *= mask
        kernel /= kernel.max()
        kernel *= self.gain

        for dy in range(-radius, radius):
            for dx in range(-radius, radius):
                i, j = cy + dy, cx + dx
                if 0 <= i < self.n and 0 <= j < self.n:
                    influence = kernel[dy + radius, dx + radius] * self.current_value
                    self.heat[i, j] = min(1.0, self.heat[i, j] + influence)
                    color = self.map_heat_to_hex(self.heat[i, j])
                    self.canvas.create_rectangle(
                        j * self.cell_size, i * self.cell_size,
                        (j + 1) * self.cell_size, (i + 1) * self.cell_size,
                        fill=color, outline=""
                    )

    def paint(self, event):
        x, y = event.x // self.cell_size, event.y // self.cell_size
        if 0 <= x < self.n and 0 <= y < self.n:
            sigma = max(0.5, self.pen_radius / 1.2)
            if self.last_x is not None:
                steps = max(abs(x - self.last_x), abs(y - self.last_y))
                for i in range(steps + 1):
                    interp_x = int(self.last_x + (x - self.last_x) * i / steps)
                    interp_y = int(self.last_y + (y - self.last_y) * i / steps)
                    self.apply_gaussian(interp_x, interp_y, radius=self.pen_radius, sigma=sigma)
            else:
                self.apply_gaussian(x, y, radius=self.pen_radius, sigma=sigma)
            self.last_x = x
            self.last_y = y

    def clear_last_point(self):
        self.last_x = None
        self.last_y = None

    def map_heat_to_hex(self, heat_val):
        rgb = self.colormap(heat_val)[:3]
        return '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)

    def draw_all(self):
        for y in range(self.n):
            for x in range(self.n):
                heat_val = self.heat[y, x]
                color = self.map_heat_to_hex(heat_val)
                self.canvas.create_rectangle(
                    x * self.cell_size, y * self.cell_size,
                    (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                    fill=color, outline=""
                )

    def reset(self):
        self.heat.fill(0)
        self.canvas.delete("all")
        self.draw_all()

    def export(self):
        print(self.heat)
        return self.heat

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    drawer = HeatMapDrawer_smooth(n=100)
    drawer.run()
    heat_matrix = drawer.heat
    print(heat_matrix)
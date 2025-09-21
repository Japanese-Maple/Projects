import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# ==== ODE solver ====
def rk4_onestep(f, dt, t0, y0):
    f1 = f(t0, y0)
    f2 = f(t0 + dt/2, y0 + dt/2 * f1)
    f3 = f(t0 + dt/2, y0 + dt/2 * f2)
    f4 = f(t0 + dt, y0 + dt * f3)
    y_output = y0 + dt/6 * (f1 + 2*f2 + 2*f3 + f4)
    return y_output

# ==== Physics ====
def three_body_problem(t, s):
    G = 1

    m1 = 1
    m2 = 10
    m3 = 1

    p1, p2, p3 = s[0], s[1], s[2]
    v1, v2, v3 = s[3], s[4], s[5]

    r12 = np.linalg.norm(p1 - p2)**3
    r13 = np.linalg.norm(p1 - p3)**3
    r23 = np.linalg.norm(p2 - p3)**3

    a1 = -G*m2*(p1 - p2)/r12 - G*m3*(p1 - p3)/r13
    a2 = -G*m1*(p2 - p1)/r12 - G*m3*(p2 - p3)/r23
    a3 = -G*m1*(p3 - p1)/r13 - G*m2*(p3 - p2)/r23

    return np.array([v1, v2, v3, a1, a2, a3])

# ==== Initial conditions ====
dt = 0.001
steps = 10000

# p1 = np.array([1, 0, 0])
# p2 = np.array([-0.1, 0, 1])
# p3 = np.array([1, 1, 1])
# v1 = np.array([1, 0, 0])
# v2 = np.array([0, 1, 1])
# v3 = np.array([0.1, 0, 0])

p1 = np.array([1, 0, 0])
p2 = np.array([-2, 0, 1])
p3 = np.array([1, 2, 1])

v1 = np.array([1, 0, 0])
v2 = np.array([1, 1, 1])
v3 = np.array([0.1, 0 , 0])

s = np.array([p1, p2, p3, v1, v2, v3])
S_3 = np.zeros((steps, 6, 3))
S_3[0] = s

s_initial = s
for i in range(steps - 1):
    s_next = rk4_onestep(three_body_problem, dt, 0, s_initial)
    S_3[i + 1] = s_next
    s_initial = s_next

# ==== Frame skipping ====
S_3_reduced = S_3[::10]
num_frames = len(S_3_reduced)

# ==== Animation setup ====
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

line1, = ax.plot([], [], [], 'red', label='p1')
line2, = ax.plot([], [], [], 'blue', label='p2')
line3, = ax.plot([], [], [], 'black', label='p3')

dot1, = ax.plot([], [], [], 'ro')   # red dot
dot2, = ax.plot([], [], [], 'bo')   # blue dot
dot3, = ax.plot([], [], [], 'ko')   # black dot

for axis in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    axis.fill = False
    axis.set_edgecolor('w')
ax.grid(False)
ax.legend()

window_radius = 3.5
trail_length = 2

trail_segments = 70   # Number of trail segments
trail_length_per_segment = 3  # Frames per segment

# Create separate trail line objects for each color
trail_lines_red = []
trail_lines_blue = []
trail_lines_black = []

for i in range(trail_segments):
    line_r, = ax.plot([], [], [], color='red', alpha=0.5*(1 - i/trail_segments))
    line_b, = ax.plot([], [], [], color='blue', alpha=0.5*(1 - i/trail_segments))
    line_k, = ax.plot([], [], [], color='black', alpha=0.5*(1 - i/trail_segments))
    trail_lines_red.append(line_r)
    trail_lines_blue.append(line_b)
    trail_lines_black.append(line_k)

def init():
    for line in (line1, line2, line3):
        line.set_data([], [])
        line.set_3d_properties([])

    for tline in trail_lines_red:
        tline.set_data([], [])
        tline.set_3d_properties([])

    for tline in trail_lines_blue:
        tline.set_data([], [])
        tline.set_3d_properties([])

    for tline in trail_lines_black:
        tline.set_data([], [])
        tline.set_3d_properties([])

    dot1.set_data([], [])
    dot1.set_3d_properties([])
    dot2.set_data([], [])
    dot2.set_3d_properties([])
    dot3.set_data([], [])
    dot3.set_3d_properties([])

    return line1, line2, line3, dot1, dot2, dot3, *trail_lines_red, *trail_lines_blue, *trail_lines_black

def update(frame):
    p1 = S_3_reduced[frame, 0]
    p2 = S_3_reduced[frame, 1]
    p3 = S_3_reduced[frame, 2]

    start = max(0, frame - trail_length)

    # Main trails
    line1.set_data(S_3_reduced[start:frame, 0, 0], S_3_reduced[start:frame, 0, 1])
    line1.set_3d_properties(S_3_reduced[start:frame, 0, 2])

    line2.set_data(S_3_reduced[start:frame, 1, 0], S_3_reduced[start:frame, 1, 1])
    line2.set_3d_properties(S_3_reduced[start:frame, 1, 2])

    line3.set_data(S_3_reduced[start:frame, 2, 0], S_3_reduced[start:frame, 2, 1])
    line3.set_3d_properties(S_3_reduced[start:frame, 2, 2])

    dot1.set_data([p1[0]], [p1[1]])
    dot1.set_3d_properties([p1[2]])
    dot2.set_data([p2[0]], [p2[1]])
    dot2.set_3d_properties([p2[2]])
    dot3.set_data([p3[0]], [p3[1]])
    dot3.set_3d_properties([p3[2]])

    # Disappearing trail segments for p1
    for i, tline in enumerate(trail_lines_red):
        seg_start = frame - (i + 1) * trail_length_per_segment
        seg_end = frame - i * trail_length_per_segment
        if seg_start < 0:
            tline.set_data([], [])
            tline.set_3d_properties([])
            continue
        x = S_3_reduced[seg_start:seg_end, 0, 0]
        y = S_3_reduced[seg_start:seg_end, 0, 1]
        z = S_3_reduced[seg_start:seg_end, 0, 2]
        tline.set_data(x, y)
        tline.set_3d_properties(z)

    # Disappearing trail segments for p2
    for i, tline in enumerate(trail_lines_blue):
        seg_start = frame - (i + 1) * trail_length_per_segment
        seg_end = frame - i * trail_length_per_segment
        if seg_start < 0:
            tline.set_data([], [])
            tline.set_3d_properties([])
            continue
        x = S_3_reduced[seg_start:seg_end, 1, 0]
        y = S_3_reduced[seg_start:seg_end, 1, 1]
        z = S_3_reduced[seg_start:seg_end, 1, 2]
        tline.set_data(x, y)
        tline.set_3d_properties(z)

    # Disappearing trail segments for p3
    for i, tline in enumerate(trail_lines_black):
        seg_start = frame - (i + 1) * trail_length_per_segment
        seg_end = frame - i * trail_length_per_segment
        if seg_start < 0:
            tline.set_data([], [])
            tline.set_3d_properties([])
            continue
        x = S_3_reduced[seg_start:seg_end, 2, 0]
        y = S_3_reduced[seg_start:seg_end, 2, 1]
        z = S_3_reduced[seg_start:seg_end, 2, 2]
        tline.set_data(x, y)
        tline.set_3d_properties(z)

    center = (p1 + p2 + p3) / 3
    ax.set_xlim(center[0] - window_radius, center[0] + window_radius)
    ax.set_ylim(center[1] - window_radius, center[1] + window_radius)
    ax.set_zlim(center[2] - window_radius, center[2] + window_radius)

    return line1, line2, line3, dot1, dot2, dot3, *trail_lines_red, *trail_lines_blue, *trail_lines_black

anim = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                     interval=10, blit=False)

# Save video (optional)
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=30)
anim.save("three_body_following.mp4", writer=writer, dpi=100)

plt.show()

import numpy as np  
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.colors import Normalize

# figure
fig, VF = plt.subplots()
VF.set_aspect(1/VF.get_data_ratio(), adjustable='box') #square the figure


# meshgrid
b=2
a=-b
c=30

x, y = np.meshgrid(np.linspace(a,b, c),
                   np.linspace(a,b, c))
z = np.cos(x)+np.sin(y)


# gradient of a function z
i, j = np.gradient(z)
i_j_magnitude=np.sqrt(i**2 + j**2)
i_norm= 1.5*i/i_j_magnitude
j_norm= 1.5*j/i_j_magnitude


# plotting 

P = plt.pcolormesh(x, y, i_j_magnitude, shading = 'gouraud', cmap ='viridis')
Q = plt.quiver(x,y,   j_norm,i_norm)

plt.colorbar(P)



#adding a 3D graph with a 2d 'projection'
b1=2*np.pi
a1=-b1
c1=60

x1, y1 = np.meshgrid(np.linspace(a1,b1, c1),
                   np.linspace(a1,b1, c1))
z1 = np.cos(x1)+np.sin(y1)

xyz = fig.add_axes([0.01, 0.25, 0.5, 0.5], projection='3d') # x-position, y-position, width, height
# Get rid of colored axes planes
# First remove fill
xyz.xaxis.pane.fill = False
xyz.yaxis.pane.fill = False
xyz.zaxis.line.set_lw(0.)
xyz.set_zticks([])
# Remove the edges
xyz.xaxis.pane.set_edgecolor('w')
xyz.yaxis.pane.set_edgecolor('w')
xyz.grid(False)

# plot the 3D function
xyz.set_box_aspect(aspect = (2*x1.max(),2*y1.max(),2*z1.max())) #set the proportions
xyz.plot_surface(x1, y1, z1, cmap = 'inferno')

# quiver/VF
i2, j2 = np.gradient(z1)
i2_j2_magnitude=np.sqrt(i2**2 + j2**2)
i2_norm= i2/i2_j2_magnitude
j2_norm= j2/i2_j2_magnitude

norm = Normalize()
colormap = mpl.cm.inferno
norm.autoscale(i2_j2_magnitude)

d=5
Q1 = xyz.quiver(x1[::d,::d], y1[::d,::d], z.min()-1, 
                
                j2_norm[::d,::d], i2_norm[::d,::d], 0, color = colormap(norm(i2_j2_magnitude[1])))

# slider for density
slider0 = fig.add_axes([0.4, 0.09, 0.4, 0.05]) # x-position, y-position, width, height
density = Slider(slider0,                 
                  '$\\mathcal{Density}$',  
                  color ='#53ff45', 
                  track_color= '#399792', 
                  valmin=1, valmax=10, 
                  valinit=1, 
                  valstep=1)

def update1(val): 
    d = density.val
    
    fig.canvas.draw_idle()

density.on_changed(update1) 





# slider for parameter 'a'
fig.subplots_adjust(bottom=0.25)

slider1 = fig.add_axes([0.4, 0.15, 0.4, 0.05]) # x-position, y-position, width, height
function = Slider(slider1,                 
                  '$\\mathcal{Parameter}$',  
                  color ='#53ff45', 
                  track_color= '#399792', 
                  valmin=-10, valmax=10, 
                  valinit=-10, 
                  valstep=0.01)



def update(val): # MAIN PART !!!
    a = function.val
    z = np.cos(x+a)+np.sin(y+a)
    
    i1, j1 = np.gradient(z)    
    i1_j1_magnitude=np.sqrt(i1**2 + j1**2)
    i1_norm= 1.5*i1/i1_j1_magnitude
    j1_norm= 1.5*j1/i1_j1_magnitude

    P.set_array(i1_j1_magnitude)
    Q.set_UVC(j1_norm, i1_norm)
    
    fig.canvas.draw_idle()
    

function.on_changed(update)



# labels
VF.set_xlabel('x', color = 'green', size = 20)
VF.set_ylabel('y', rotation = 0, color = 'green', size = 20)
VF.set_title('$\\mathcal{z = cos(x+a)+sin(y+a)}$', y = 1.03, color = '#2C63B8', size = 30)


# result
plt.show()

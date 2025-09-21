import numpy as np
import matplotlib.pyplot as plt

t = np.load('train_loss_FNO.npy')
v = np.load('val_loss_FNO.npy')
x = np.arange(len(t))
fig, ax = plt.subplots(2, 1, figsize = (21, 12), dpi = 100)
for i in range(2):
    ax[i].plot(t, linestyle = '-',  label = 'Train Loss', color = 'red')
    ax[i].plot(v, linestyle = '--', label = "Validation Loss", color = 'blue')
    ax[i].scatter(x,t, c='r')
    ax[i].scatter(x,v, c='b')

    ax[i].set_xlabel('Epochs')
    ax[i].set_ylabel('Loss Value')    
    ax[i].legend()
    ax[i].grid(True)

ax[0].set_title("FNO Loss Plot regular Scale")

ax[1].set_yscale('log')
ax[1].set_title("FNO Loss Plot log Scale")

plt.savefig('FNO_loss_plot.pdf')
plt.show()

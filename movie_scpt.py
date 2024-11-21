
# Movie script ...
# Parameters:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_density(x1,x2,p,file_name='FPE_movie',file_type='gif',fps=50):
    ### animation

    fig = plt.figure(figsize=plt.figaspect(1/2))
    ax = fig.add_subplot(projection='3d')

    surf = ax.plot_surface(*np.meshgrid(x1,x2), p[:,:,0], 
                           cmap='jet', rcount=200, ccount=200)

    ax.set_zlim([0,1.1*np.max(p)])
    ax.autoscale(False)
    ax.set(xlabel='x_1', ylabel='x_2', zlabel='P(x_1,x_2,t)')

    def update(i):
        global surf
        surf.remove()
        surf = ax.plot_surface(*np.meshgrid(x1,x2), p[:,:,i], cmap='jet')
        return surf

    anim = FuncAnimation(fig, update, frames=range(p.shape[-1]))
    plt.tight_layout()
    plt.show()

    writer = PillowWriter(fps=fps, bitrate=1800)
    anim.save(file_name+'.'+file_type, writer=writer)
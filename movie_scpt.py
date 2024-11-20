
# Movie script ...
# Parameters:
import numpy as np
import matplotlib.pyplot as plt

skip_frame = 2 * 4
from_top = True
writeVids = False
equal_axis = True
plt.figure('units','pixels','position',np.array([0,0,1280,720]))
set(plt.gcf(),'Color','white')
plt.surf(x2,x1,p[:,:,0],'edgecolor','none')
plt.axis(np.array([x1(1),x1(end()),x2(1),x2(end()),- 0.05,0.2]))
set(plt.gca(),'nextplot','replacechildren','Visible','on')
if equal_axis:
    plt.axis('equal')

if writeVids:
    myVideo = VideoWriter('FPE_movie.avi')
    myVideo.FrameRate = 50
    myVideo.Quality = 75
    open_(myVideo)

for j in np.arange(2,len(t)+skip_frame,skip_frame).reshape(-1):
    #drawnow
    plt.surf(x2,x1,p[:,:,j],'edgecolor','none')
    if from_top:
        az = 0
        el = 90
        view(az,el)
    if writeVids:
        writeVideo(myVideo,getframe(gca))
    else:
        drawnow
    #pause(.1)

if writeVids:
    close_(myVideo)

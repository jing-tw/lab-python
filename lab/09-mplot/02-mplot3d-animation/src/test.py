import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # create a 3d grpah and add it to the fig

for i in range(100):
    if i%2 == 0:
        bb = 100
    else:
        bb = -100

    x = [1, bb]
    y = [1, bb]
    z = [1, bb]

    ax.clear()  # clean the screen
    ax.set(xlim=(-500, 500), ylim=(-500, 500), zlim=(-500, 500)) # setup the scale
    ax.set_xlabel('$X$', fontsize=20) # update the axis label
    ax.set_ylabel("$Y$ axis label")
    ax.set_zlabel("$Z$ axis label")
    
    ax.plot(x, y, z, label='line')
    ax.legend() # show the legend
    
    #fig.canvas.draw()
    
    plt.show(block=False)
    plt.pause(0.1)


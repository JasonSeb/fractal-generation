"""
# Author  : Jason Connie
# Created : July 2022

This code generates an animation zooming into a region of the Mandelbrot set.
The animation is saved as an animated png, titled 'mandelbrot.png'.

NOTES: As they are generated, the individual frames of the animation are saved in a folder 
       titled 'AnimatedFrames'. If you choose to make an animation of a 4 times zoom and find 
       it agreeable, but wish to zoom further, running the program again with a higher zoom 
       will not redundantly produce frames that have already been made. In this way deeper and 
       deeper zooms can be generated without having to start from scratch each time.
       
       If you are starting a new animation from scratch centred on a completely new point, or 
       are still dealing with the same point but have simply altered other variables like delta 
       (which influences the smoothness of the animation), you will have to manually delete the 
       'AnimatedFrames' folder beforehand. 
       
       Fractal generation is computationally expensive at the best of times, and this code 
       is no exception. Animating a 2 times zoom-in seems to require a minimum of 40 frames 
       for things to not look too jumpy. An equal amount of frames will be needed each time 
       you zoom in by a factor of 2. Runtimes will take minutes, and even longer if you are 
       trying to generate a deep zoom all at once. 
       
       If you wish to investigate the Mandelbrot set dynamically and in real time, the file 
       'Mandelbrot_interactive.py' might be of interest to you. As you zoom in and explore, 
       the interactive program tells you the zoom you have applied as well as the point you 
       are centred on. As such,'Mandelbrot_interactive.py' can be used to find good candidate 
       coordinates to animate.
       
"""

# We import the necessary modules
import numpy as np
import concurrent.futures
import os
from numba      import jit
from itertools  import repeat
from matplotlib import pyplot as plt
from apng       import APNG
# import pathlib


##################################################################################
## Methods for Mandelbrot fractal generation, and for generating the animation ###
##################################################################################

""" Method to return the number of iterations needed for an inputted complex value c to diverge
    If the inputted c does not diverge by the inputted threshold, we return 0 """
@jit
def mandel(c, threshold):
    z = 0 + 1j*0
    for n in range(threshold):
        if (z.real*z.real + z.imag*z.imag)>4:   #if z diverges
            return n
        z = (z * z) + c
    return 0




""" Method to generate the colour values for multiple points 
    To be called by an executor in parallel """
@jit
def mandelbrot_for_parallel(xmin, xmax, ymin, ymax, threshold, yrange):
    re = np.linspace(xmin, xmax, resolution)
    im = np.linspace(ymin, ymax, resolution)

    k_len      = resolution*len(yrange)
    point_vals = np.array([0.0]*k_len)
    k = 0

    for y in yrange:
        for x in range(resolution):
            point_vals[k] = mandel(re[x] + 1j*im[y], threshold)
            k+=1

    return point_vals




""" Method to generate a frame of our animation """
def animate(i):    
    global xmin, xmax, ymin, ymax, threshold, delta
    
    new_xmin = xmin + 0.5*(xmax-xmin)*(1-1/2**(delta*i))
    new_xmax = xmax - 0.5*(xmax-xmin)*(1-1/2**(delta*i))
    new_ymin = ymin + 0.5*(ymax-ymin)*(1-1/2**(delta*i))
    new_ymax = ymax - 0.5*(ymax-ymin)*(1-1/2**(delta*i))
    
    new_threshold = threshold + int(delta*i*50)
    
    # Recalculate the plot, and display
    with concurrent.futures.ProcessPoolExecutor() as executor:
        parrays = executor.map(mandelbrot_for_parallel, repeat(new_xmin), repeat(new_xmax), repeat(new_ymin), repeat(new_ymax), repeat(new_threshold), yranges)

        complete_parray = []

        for parray in parrays:
            complete_parray = np.concatenate([complete_parray, parray])

        complete_parray = np.reshape(complete_parray, [resolution, resolution])
    
    img = ax.imshow(complete_parray, interpolation='bicubic', cmap=color)
    
    return [img]





####################################################
# Main method, called to generate the animated gif #
####################################################

if __name__=="__main__":
    global resolution, threshold, color, xmin, xmax, ymin, ymax, num_processes, yranges

    ##### NB: Here we set the zoom at which our animation terminates,
    #####     and set the x and y coordinates to zoom in on. Adjust as desired!
    zoom =  2**16
    p_re =  0.360240443437614363236 
    p_im = -0.641313061064803174860


    # We set the resolution, the threshold (the initial number of iterations used to see if a point diverges)
    # and delta (the factor that dictates the scale difference between frames, and thus how smooth the animation is)
    resolution = 2000
    threshold  = 70
    delta      = 0.025    # Smaller deltas give smoother animation, though at the price of being computationally more expensive
    color      = 'magma' 
    
    
    # Setting the initial x and y boundaries, centred at the chosen point
    xmin = p_re - 1.5
    xmax = p_re + 1.5
    ymin = p_im - 1.5
    ymax = p_im + 1.5

    
    
    # We split up the y_ranges, to be acted on in parallel. We also set the number of frames used in the animation, as dependent on the zoom
    num_processes = int(1.5*os.cpu_count())
    yranges       = np.array_split(range(resolution), num_processes)
    num_frames    = int(np.log2(zoom)/delta)
    
    plt.style.use('dark_background')    
    fig = plt.figure(figsize=(resolution/80, resolution/80))
    ax  = plt.axes()
    plt.axis('off')
    plt.tight_layout()
    
    
    
    # If the AnimatedFrames folder doesn't already exist, we make it
    if not os.path.exists('./AnimatedFrames'):
        os.makedirs('./AnimatedFrames')
   
    frames = [None]*num_frames

    for i in range(num_frames):
        file_name = f"AnimatedFrames/frame{i}.png"
        
        if not os.path.exists(file_name): # If the ith frame does not exist, only then do we generate it
            animate(i)
            plt.savefig(file_name)
        
        frames[i] = file_name
    
    # Generating the animated PNG of the fractal zoom. Adjusting delay will alter the speed of the PNG
    APNG.from_files(frames, delay=40).save('mandelbrot_zoom.png')

    


"""
# Author  : Jason Connie
# Created : July 2022

This code generates an interactive plot of the Mandelbrot set.
  - Left clicking on a point applies a two times zoom, centred on the clicked point.
  - Right clicking on a point applies a two times zoom-out, similarly centred on the clicked point.


NOTE: Adjustments have been made, but this code is derived from a code sample originally available from
      - https://stackoverflow.com/questions/60053881/mandelbrot-set-gets-blurry-at-around-247-zoom
      The biggest adjustment is that I've PARALLELISED the code to make deeper zooms more efficient.
      As in the original code, there are slight inaccuricies due to floating point errors. Details get 
      blurred when zooming beyond 2**(-45), which is approximately a zoom of ten trillion.
      
"""

# We import the necessary modules
import numpy as np
import concurrent.futures
import os
from numba      import jit
from itertools  import repeat
from matplotlib import pyplot as plt





##################################################################################
# Methods for Mandelbrot fractal generation, and for responding to a click event #
##################################################################################

""" Method to return the number of iterations needed for an inputted complex value c to diverge
    If the inputted c does not diverge by the threshold, we return 0

    @jit optimizes our code. The @ indicates that jit is a decorator 
    (NOTE: Decorators are really cool!) """
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




""" Method to define what is done when a point on the plot is clicked 
      - Left clicking on a point applies a two times zoom, centred on the clicked point.
      - Right clicking on a point applies a two times zoom-out, similarly centred on the clicked point. """
def onclick(event):
    global zoom, xmax, xmin, ymax, ymin, threshold

    # When someone has clicked, we solve for the new zoom and threshold values
    if event.button == 1:
        zoom = zoom*2
        threshold = threshold + add_thresh
    elif event.button == 3:
        zoom = int(zoom/2)
        if threshold > add_thresh:
            threshold = threshold - add_thresh

    # Get the coordinates of the point the mouse has clicked on
    posx = xmin + (xmax-xmin)*event.xdata/resolution
    posy = ymin + (ymax-ymin)*event.ydata/resolution

    # Print the current central point and zoom value
    print("The current zoom is ", zoom, " and the central point is (",posx,",",posy,")",sep="")
    
    # Adjust the scales of our real and imaginary axes, so that the point clicked on serves as the centre point of the new plot
    xmin = posx - 1.5/zoom
    xmax = posx + 1.5/zoom
    ymin = posy - 1.5/zoom
    ymax = posy + 1.5/zoom
    
    # Recalculate the plot in parallel, and display
    with concurrent.futures.ProcessPoolExecutor() as executor:
        parrays = executor.map(mandelbrot_for_parallel, repeat(xmin), repeat(xmax), repeat(ymin), repeat(ymax), repeat(threshold), yranges)

        complete_parray = []

        for parray in parrays:
            complete_parray = np.concatenate([complete_parray, parray])

        complete_parray = np.reshape(complete_parray, [resolution, resolution])

    plt.imshow(complete_parray, cmap=color)
    plt.draw()





#############################################
# Main method, called to run the whole show #
#############################################

if __name__=="__main__":
    # We set the resolution, threshold, initial zoom and the color map to be used
    resolution = 2800  # Can be adjusted, but the higher the resolution the longer the render time
    threshold  = 70    # The initial number of iterations used to see if a point diverges or not
    add_thresh = 50    # At each 'zoom' we will add to the threshold, to get more finely grained detail
    zoom       = 1
    color      = 'magma'
    
    
    # We set the the x and y boundaries of our initial plot
    xmin = -2
    xmax =  1
    ymin = -1.5
    ymax =  1.5
    
    
    # We split the y-axis into different ranges, to be acted on in parallel
    num_processes = int(1.5*os.cpu_count())
    yranges       = np.array_split(range(resolution), num_processes)


    # We solve for the first plot, in parallel like the rest
    with concurrent.futures.ProcessPoolExecutor() as executor:
        parrays = executor.map(mandelbrot_for_parallel, repeat(xmin), repeat(xmax), repeat(ymin), repeat(ymax), repeat(threshold), yranges)    
        complete_parray = []

        for parray in parrays:
            complete_parray = np.concatenate([complete_parray, parray])

        complete_parray = np.reshape(complete_parray, [resolution, resolution])
    
    
    # We make the plot with our desired stylistic choices
    plt.style.use('dark_background')
    plt.connect('button_press_event', onclick)
    plt.imshow(complete_parray, cmap=color)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


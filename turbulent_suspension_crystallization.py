#!/usr/env/python

"""
isotropic_turbulent_suspension_crystallization.py

Example of a continuous-time, stochastic, pair-based cellular automaton model, 
which simulates the uninhibited crystallization of a mineral

Written by Ryan Stoner April 2016, modified from Greg Tucker, February 2015
"""

import time
import matplotlib
import numpy as np
from pylab import figure, plot, show
from numpy import where, zeros, mean
from landlab import RasterModelGrid
from landlab.ca.celllab_cts import Transition, CAPlotter
from landlab.ca.oriented_raster_cts import OrientedRasterCTS


def setup_transition_list():
    """
    Creates and returns a list of Transition() objects to represent state
    transitions for a biased random walk, in which the rate of downward
    motion is greater than the rate in the other three directions.
    
    Parameters
    ----------
    (none)
    
    Returns
    -------
    xn_list : list of Transition objects
        List of objects that encode information about the link-state transitions.
    
    Notes
    -----
    State 0 represents fluid, state 1 represents a particle (to be
    crystallized), state 2 is the crystal. Trying to recreate growth patterns
    of uninhibited crystal growth similar to native copper or hoar frost.
    
    The states and transitions are as follows:

    Pair state      Transition to       Process             Rate (cells/s)
    ==========      =============       =======             ==============
    0 (0-0)         (none)              -                   -
    1 (0-1)         2 (1-0)             left motion         10.0
    2 (1-0)         1 (0-1)             right motion        10.0
    3 (1-1)         (none)              -                   -
    4 (0-0)         (none)              -                   -
    5 (0-1)         2 (1-0)             down motion         10.55
    6 (1-0)         1 (0-1)             up motion            9.45
    7 (1-1)         (none)              -                   -
    
    8 (2-0)         (none)              -                   -
    9 (2-1)         (2-2)               left/crystal.      (n.a.)
    10(1-2)         (2-2)               right/crystal.     (n.a.)
    11(0-2)         (none)              -                   -
    12(2-2)         (none)              -                   -
    
    13(2-1)         (2-2)               up/crystal.        (n.a.)
    14(1-2)         (2-2)               down/crystal.      (n.a.)
    """
    
    # Create an empty transition list
    xn_list = []
    
    # Append four transitions to the list.
    # Note that the arguments to the Transition() object constructor are:
    #  - Tuple representing starting pair state
    #    (left cell, right cell, orientation [0=horizontal])
    #  - Tuple representing new pair state
    #    (bottom cell, top cell, orientation [1=vertical])
    #  - Transition rate (cells per time step, in this case 1 sec)
    #  - Name for transition
    xn_list.append( Transition((0,1,0), (1,0,0), 50., 'left motion') )
    xn_list.append( Transition((1,0,0), (0,1,0), 50., 'right motion') )
    xn_list.append( Transition((0,1,1), (1,0,1), 50., 'down motion') )
    xn_list.append( Transition((1,0,1), (0,1,1), 50., 'up motion') )
    xn_list.append( Transition((1,2,1), (2,2,1), 100., 'down capture') )
    xn_list.append( Transition((2,1,1), (2,2,1), 100., 'up capture') )
    xn_list.append( Transition((1,2,0), (2,2,1), 100., 'left capture') )
    xn_list.append( Transition((2,1,0), (2,2,1), 100., 'right capture') )
    return xn_list
    
    
def main():
    
    # INITIALIZE

    # User-defined parameters
    nr = 100  # number of rows in grid
    nc = 64  # number of columns in grid
    plot_interval = 0.5   # time interval for plotting, sec
    run_duration = 30.0   # duration of run, sec
    report_interval = 10.0  # report interval, in real-time seconds
  
    # Remember the clock time, and calculate when we next want to report
    # progress.
    current_real_time = time.time()
    next_report = current_real_time + report_interval

    # Create grid
    mg = RasterModelGrid(nr, nc, 1.0)
    
    # Make the boundaries be walls
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
    
    # Set up the states and pair transitions.
    ns_dict = { 0 : 'fluid', 1 : 'particle', 2: 'crystal' }
    xn_list = setup_transition_list()

    # Create the node-state array and attach it to the grid
    node_state_grid = mg.add_zeros('node', 'node_state_map', dtype=int)
    
    # Initialize the node-state array: here, the initial condition is a pile of
    # resting moving particles at the bottom of a container. Seed crystal is 
    bottom_rows = where(mg.node_y<0.1*nr)[0]
    node_state_grid[bottom_rows] = 1
    
    
    seed_crystal_rows = where(mg.node_y>0.5*nr)[0]
    node_state_grid[seed_crystal_rows] = 2
    
    # For visual display purposes, set all boundary nodes to fluid
    node_state_grid[mg.closed_boundary_nodes] = 0
    
    # Create the CA model
    ca = OrientedRasterCTS(mg, ns_dict, xn_list, node_state_grid)
    
    # Hex colors
    grain = '#5F594D'
    fluid = '#D0E4F2'
    crystal = '#000000'
    clist = [fluid,grain,crystal]
    my_cmap = matplotlib.colors.ListedColormap(clist)

    # Create a CAPlotter object for handling screen display
    ca_plotter = CAPlotter(ca, cmap=my_cmap)
    
    # Plot the initial grid
    ca_plotter.update_plot()

    # RUN
    current_time = 0.0
    while current_time < run_duration:
        
        # Once in a while, print out simulation and real time to let the user
        # know that the sim is running ok
        current_real_time = time.time()
        if current_real_time >= next_report:
            print 'Current sim time',current_time,'(',100*current_time/run_duration,'%)'
            next_report = current_real_time + report_interval
        
        # Run the model forward in time until the next output step
        ca.run(current_time+plot_interval, ca.node_state, 
               plot_each_transition=False)
        current_time += plot_interval
        
        # Plot the current grid
        ca_plotter.update_plot()
        

    # FINALIZE

    # Plot
    ca_plotter.finalize()
    
    # Calculate concentration profile
    c = zeros(nr)
    for r in range(nr):
        c[r] = mean(node_state_grid[r*nc:(r+1)*nc])
        
    figure(2)
    plot(c, range(nr), 'o')
    show()

# If user runs this file, activate the main() function
if __name__ == "__main__":
    main()

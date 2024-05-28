# -*- coding: utf-8 -*-

# future import statements
from __future__ import print_function
from __future__ import division

# version information
__project__ = "EXPLORE"
__author__  = "ACRI-ST"
__modifiers__ = '$Author: N. Cox $'
__date__ = '$Date: 2022-07-15 $'
__version__ = '$Rev: 2.0 $'
__license__ = '$Apache 2.0 $'

import os
from load_cube import load_cube
import numpy as np


def initialise(cubename=None):
 
    """ path to local hdf5 data cube (testing) """
    #hdf5file = os.path.join("/home/nick/Sandbox/cube/", "stilism_cube_2.h5")

    """ set global variables cube 1 (25 pc) """
    hdf5file25 = os.path.join(os.environ.get('SERVICE_APP_DATA'), "explore_cube_density_values_025pc_v1.h5")
    global headers25, cube25, axes25, min_axes25, max_axes25, step25, hw25, points25, s25
    headers25, cube25, axes25, min_axes25, max_axes25, step25, hw25, points25, s25 = load_cube(hdf5file25)

    """ set global variables cube 2 (50 pc) """
    hdf5file50 = os.path.join(os.environ.get('SERVICE_APP_DATA'), "explore_cube_density_values_050pc_v1.h5")
    global headers50, cube50, axes50, min_axes50, max_axes50, step50, hw50, points50, s50
    headers50, cube50, axes50, min_axes50, max_axes50, step50, hw50, points50, s50 = load_cube(hdf5file50)

    """ read the density and extinction error cubes (one resolution only) """
    hdf5file_densityerror = os.path.join(os.environ.get('SERVICE_APP_DATA'), "explore_cube_density_errors_050pc_v1.h5")
    hdf5file_extincterror = os.path.join(os.environ.get('SERVICE_APP_DATA'), "explore_cube_extinct_errors_050pc_v1.h5")

    global headers_errdens, cube_errdens, axes_errdens, min_axes_errdens, max_axes_errdens, step_errdens, hw_errdens, points_errdens, s_errdens
    headers_errdens, cube_errdens, axes_errdens, min_axes_errdens, max_axes_errdens, step_errdens, hw_errdens, points_errdens, s_errdens = load_cube(hdf5file_densityerror)

    global headers_errext, cube_errext, axes_errext, min_axes_errext, max_axes_errext, step_errext, hw_errext, points_errext, s_errext
    headers_errext, cube_errext, axes_errext, min_axes_errext, max_axes_errext, step_errext, hw_errext, points_errext, s_errext = load_cube(hdf5file_extincterror) 


    """ set global variables cube v2-10pc """
    cubeV2_10pc = os.path.join(os.environ.get('SERVICE_APP_DATA'), "explore_cube_density_values_010pc_v2.h5")
    global headers_v2_10, cube_v2_10, axes_v2_10, min_axes_v2_10, max_axes_v2_10, step_v2_10, hw_v2_10, points_v2_10, s_v2_10
    headers_v2_10, cube_v2_10, axes_v2_10, min_axes_v2_10, max_axes_v2_10, step_v2_10, hw_v2_10, points_v2_10, s_v2_10 = load_cube(cubeV2_10pc)

    """ set global variables cube v2-25pc """
    cubeV2_25pc = os.path.join(os.environ.get('SERVICE_APP_DATA'), "explore_cube_density_values_025pc_v2.h5")
    global headers_v2_25, cube_v2_25, axes_v2_25, min_axes_v2_25, max_axes_v2_25, step_v2_25, hw_v2_25, points_v2_25, s_v2_25
    headers_v2_25, cube_v2_25, axes_v2_25, min_axes_v2_25, max_axes_v2_25, step_v2_25, hw_v2_25, points_v2_25, s_v2_25 = load_cube(cubeV2_25pc)

    """ set global variables cube v2-50pc """
    cubeV2_50pc = os.path.join(os.environ.get('SERVICE_APP_DATA'), "explore_cube_density_values_050pc_v2.h5")
    global headers_v2_50, cube_v2_50, axes_v2_50, min_axes_v2_50, max_axes_v2_50, step_v2_50, hw_v2_50, points_v2_50, s_v2_50
    headers_v2_50, cube_v2_50, axes_v2_50, min_axes_v2_50, max_axes_v2_50, step_v2_50, hw_v2_50, points_v2_50, s_v2_50 = load_cube(cubeV2_50pc)

    """ read the density and extinction error cubes V2 (one resolution only) """
    cubeV2_densityerror = os.path.join(os.environ.get('SERVICE_APP_DATA'), "explore_cube_density_errors_050pc_v2.h5")
    cubeV2_extincterror = os.path.join(os.environ.get('SERVICE_APP_DATA'), "explore_cube_extinct_errors_050pc_v2.h5")

    global headers_errdens2, cube_errdens2, axes_errdens2, min_axes_errdens2, max_axes_errdens2, step_errdens2, hw_errdens2, points_errdens2, s_errdens2
    headers_errdens2, cube_errdens2, axes_errdens2, min_axes_errdens2, max_axes_errdens2, step_errdens2, hw_errdens2, points_errdens2, s_errdens2 = load_cube(cubeV2_densityerror)

    global headers_errext2, cube_errext2, axes_errext2, min_axes_errext2, max_axes_errext2, step_errext2, hw_errext2, points_errext2, s_errext2
    headers_errext2, cube_errext2, axes_errext2, min_axes_errext2, max_axes_errext2, step_errext2, hw_errext2, points_errext2, s_errext2 = load_cube(cubeV2_extincterror) 

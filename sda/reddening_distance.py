# -*- coding: utf-8 -*-

# future import statements
from __future__ import print_function
from __future__ import division

# version information
__project__ = "EXPLORE"
__author__  = "ACRI-ST"
__modifiers__ = '$Author: N. Cox $'
__date__ = '$Date: 2021-10-12 $'
__version__ = '$Rev: 1.0 $'
__license__ = '$Apache 2.0 $'

import numpy as np
import scipy.interpolate as spi
from astropy import units as u
from astropy.coordinates import SkyCoord


def reddening_distance(sc, target_distance, cube, axes, max_axes, step_pc=5, error=False):
    """Calculate cummulative extinction in a given direction and distance from the Sun.

    Args:
        sc: SkyCoord object
        target_distance (real): distance in parsec (if target_distance==None go to end of cube)

    Kwargs:
        step_pc (int): Incremental distance in parsec

    Returns:
        array: Extinction A(5500) value obtained with integral of linear extrapolation (interpolated at target_distance).

    """
    
    sc1=SkyCoord(sc, distance = 1 * u.pc)

    coords_xyz = sc1.transform_to('galactic').represent_as('cartesian').get_xyz().value

    # Find the number of parsec I can calculate before go out the cube (exclude divide by 0)
    not0 = np.where(coords_xyz != 0)

    max_pc = np.amin(
            np.abs( np.take(max_axes, not0) / np.take(coords_xyz, not0) ) )

    # Calculate all coordinates to interpolate (use step_pc)
    distances = np.arange(0, max_pc, step_pc)    

    sc2 = SkyCoord(
        sc,
        distance=distances)

    sc2 = sc2.transform_to('galactic').represent_as('cartesian')
    coords_xyz = np.array([coord.get_xyz().value for coord in sc2])

    # if target_distance == None:
    #     print(coords_xyz[-1])
    #
    # else:
    #     print(coords_xyz)
    
    # linear interpolation with coordinates
    interpolation = spi.interpn(
        axes,
        cube,
        coords_xyz,
        method='linear'
    )

    xvalues = np.arange(0, len(interpolation) * step_pc, step_pc)

    if error == True:
        yvalues = interpolation
    else:
        yvalues = np.nancumsum(interpolation) * step_pc

    if (target_distance == None) or (target_distance >= np.nanmax(xvalues)):

        distance = xvalues[-1]
        extinction = yvalues[-1]

    else:
        distance = target_distance
        f = spi.interp1d(xvalues, yvalues)
        extinction = f(target_distance)

    return (
        distance, 
        np.around(extinction, decimals=5),
        )

"""
from load_cube import load_cube
sc = SkyCoord.from_name("HD183143", frame='icrs')
step_pc = 5.0
target_distance=223.0
headers50, cube50, axes50, min_axes50, max_axes50, step50, hw50, points50, s50 = load_cube("/home/ncox/Projects/_data/gaia/explore_cube_density_values_050pc_v1.h5")
dist, ext = reddening_distance(sc, target_distance=target_distance, cube=cube50, axes=axes50, max_axes=max_axes50, step_pc=step_pc)
print(dist, ext)
"""
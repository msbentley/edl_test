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


def reddening_distance_bin(sc, d_bins, cube, axes, max_axes, step_pc=5, error=False):
    """Calculate cumulative extinction in a given direction (ra,dec) between lower and upper bound distance.

    Args:
        sc: SkyCoord object
        d_bin (list): list of distances between which the integrated extinction is returned

    Kwargs:
        step_pc (int): Incremental distance in parsec

    Returns:
        array: Integrated extinction A(5500) values obtained between consecutive values in the list

    """

    # check that d_bins has 2 elements.
    
    if len(d_bins) < 2:
        return (None,None)
    
    # make sure the list is sorted and turned into array   
    d_bins = np.array(d_bins)
    d_bins.sort()

    #print(d_bins)

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

    ## remove any distances > max_pc
    idx=np.where(d_bins>max_pc)
    d_bins[idx] = xvalues[-1]

    #print(d_bins)
    #print(max_pc)

    ##interpolation function. input array, output array.
    f = spi.interp1d(xvalues, yvalues)
    ext_bins = f(d_bins)

    extinctions = []
    for d in np.arange(len(ext_bins)-1):
            extinctions.append(  ext_bins[d+1] - ext_bins[d] )
                               
    return (
        d_bins,
        np.around(extinctions, decimals=5),
        )


"""
from load_cube import load_cube
sc = SkyCoord.from_name("HD183143", frame='icrs')
step_pc = 5.0
d_lower=127.0
d_upper=312.5
headers50, cube50, axes50, min_axes50, max_axes50, step50, hw50, points50, s50 = load_cube("/home/ncox/Projects/_data/gaia/explore_cube_density_values_050pc_v1.h5")
ext = reddening_distance_bin(sc, d_bins=[0,100,200], cube=cube50, axes=axes50, max_axes=max_axes50, step_pc=step_pc)
print(ext)
"""

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

from reddening_distance_bin import reddening_distance_bin
from load_cube import load_cube

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

## sky map integrated extinction

headers, cube, axes, min_axes, max_axes, step, hw, points, s = load_cube("/home/ncox/Projects/_data/gaia/explore_cube_density_values_050pc_v1.h5")

sc = SkyCoord.from_name("Name Cha III", frame='galactic')
step_pc = 5.0
dmin=0.0
dmax=1000.0
size=5.0
res=0.02
lon=sc.l.value
lat=sc.b.value
frame='galactic'

# create grid of coordinates
lon_grid = np.arange(lon-size/2,lon+size/2,res)
lat_grid = np.arange(lat-size/2,lat+size/2,res)
extmap = np.zeros([len(lon_grid),len(lat_grid)])

for i in range(len(lon_grid)):
        for j in range(len(lat_grid)):
                    sc=SkyCoord(lon_grid[i]*u.deg, lat_grid[j]*u.deg, frame=frame)
                    bins, ext = reddening_distance_bin(sc, d_bins=[dmin,dmax], cube=cube, axes=axes, max_axes=max_axes, step_pc=5)
                    extmap[i,j] = ext
        print(i)

## this is quite slow. better to pre-compute a grid (l,b,d) !?


print('extmap created')
## plot X,Y,Z contour plot


fig = go.Figure(data=
    go.Contour(
        z=np.log10(extmap+1.0e-7),
        x=lon_grid,
        y=lat_grid,
        name='Integrated extinction sky map',
        colorscale='Cividis_r', #'Inferno_r', #
        colorbar=dict(
            title='A(550nm) mag (log10)',
            titleside="right",
            ),
        connectgaps=True, 
        line_smoothing=0.85,
        hoverinfo='all',
        #hovertemplate="x: %{x:$.1f}, y: %{y:$.1f}, z: %{z:$.1f}",
        # zmin=-4,
        # zmax=-1.5,
        # contours=dict(
        #     start=-3.5,
        #     end=-2,
        #     size=0.5,
        # ),
        contours_coloring='heatmap', # can also be 'lines', or 'none',
    ),
    layout_xaxis_range=[np.nanmin(lon_grid),np.nanmax(lon_grid)],
    layout_yaxis_range=[np.nanmin(lat_grid),np.nanmax(lat_grid)],
    )
    
fig.show()

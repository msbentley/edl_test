# -*- coding: utf-8 -*-

# future import statements
from __future__ import print_function
from __future__ import division

# version information
__project__ = "EXPLORE"
__author__  = "ACRI-ST"
__modifiers__ = '$Author: N. Cox $'
__date__ = '$Date: 2023-04-12 $'
__version__ = '$Rev: 2.0 $'
__license__ = '$Apache 2.0 $'

import pandas as pd
import numpy as np
import pandas as pd
import json
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

import globals

from sda import sda
from reddening_distance_bin import reddening_distance_bin

from flask import Flask
from flask_caching import Cache

CACHE_CONFIG = {
    'DEBUG': True,
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': '/tmp/',
    'CACHE_DEFAULT_TIMEOUT': 600,
    'CACHE_THRESHOLD': 10, #max number of items the cache will store
}
cache = Cache()
cache.init_app(sda.server, config=CACHE_CONFIG)
""" layout figures tab 'integrated extinction map' """

fig_layout_extmap = dict(
        margin={'l': 5, 'b': 5, 't': 5, 'r': 5},
        legend={'x': 0.8, 'y': 0.1},
        hovermode='closest',
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        yaxis=dict(
            scaleanchor='x',
            scaleratio=1,
        )
    )

# create data for first map:

fig_extmap = go.Figure(data =
     go.Contour(
        z=np.log10([[5,5,5],[6,6,6],[7,7,7]]),
        name='extinction map',
        colorscale='Inferno_r', #'Cividis_r',
        colorbar=dict(
            title='A(550nm) mag/pc (log10)',
            titleside="right",
            ),
        connectgaps=True, 
        line_smoothing=0.85,
        hoverinfo='all',
        #hovertemplate="x: %{x:$.1f}, y: %{y:$.1f}, z: %{z:$.1f}",
        # zmin=-3.5,
        # zmax=-2,
        # contours=dict(
        #     start=-3.5,
        #     end=-2,
        #     size=0.5,
        # ),
        contours_coloring='heatmap', # can also be 'lines', or 'none',
    ),
    layout=fig_layout_extmap)

#fig2d.update_layout(yaxis_range=[-5000,5000], xaxis_range=[-5000,5000], yaxis=dict(scaleanchor='x'))
#fig2d['layout']['yaxis']['scaleanchor']='x'
fig_extmap.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
)

def content_extmap():
    content_extmap = html.Div([
        #dcc.Store(id='memory'),
        html.Br(),
        # Input Frame
        html.Div([
            html.H5("Inputs"),
            dcc.RadioItems(
                    id='extmap-frame',
                    options=[
                        {'label': 'Galactic coordinates (l,b)', 'value': "galactic"},
                        {'label': 'Equatorial coordinates (ra,dec)', 'value': "icrs"},
                    ],
                    value = "galactic",
            ),
            html.Br(),
            html.Label("ra/l (degrees)"),
            dcc.Input(id='extmap-lon', type='number', placeholder="[0,360]", size='10'),
            html.Br(),
            html.Label("dec/b (degrees)"),
            dcc.Input(id='extmap-lat', type='number', placeholder="[-90,90]", size='10'),
            html.Br(),
            html.Label("d(min) (pc)"),
            dcc.Input(id='extmap-dmin', type='number', placeholder="[0.0...]", size='10'),
            html.Br(),
            html.Label("d(max) (pc)"),
            dcc.Input(id='extmap-dmax', type='number', placeholder="[0.0...]", size='10'),
            html.Br(),
            html.Label("map size (degrees)"),
            dcc.Input(id='extmap-size', type='number', placeholder="[0.1,10]", size='10'),
            html.Br(),
            html.Label("map resolution (degrees)"),
            dcc.Input(id='extmap-res', type='number', placeholder="[0.01..1]", size='10'),
            html.Br(),
            html.H6("Note that size/resolution shall be less than 1000"),
            html.Br(),
            html.Button(id='extmap-submit-button', n_clicks=0, children='Submit', style={'margin-left':'10px'}),
            html.Br(),
        ], id='input', style={'width':'100%', 'float': 'top', 'margin':'10px'}),
        dcc.Loading(
                id='load-extmap',
                type='circle',
                fullscreen=False,
                children=[
                    html.Div(
                            id='tab-extmap-content', 
                            children=[],
                    ),
                ],
            ),
        # download 
        html.Div([  
            html.Button(id='save-extmap', n_clicks=0, children='Export map', style={'margin-left':'10px'}),
            dcc.Download(id='download-extmap-csv'), 
        ]),
        # Output Frame
        #html.Div(id='logmap', children=[]),
        html.Div([
            html.Hr(),
            html.Div(id='extmap-msg', children=[]),
            html.Hr(),
            dcc.Graph(id='extmap-graph', figure=fig_extmap, config={"doubleClick": "reset"}),
        ], id='output', style={'width': '100%', 'margin':'10px', 'float': 'bottom'} ), 

    ], id='extmap', style={'width':'100%', 'display':'inline-block', 'margin':'auto', 'padding-left':'10%', 'padding-right':'10%'})

    return content_extmap


""" functions """

def update_planar(X,Y,Z):
    newfig = go.Figure(data=
    go.Contour(
        z=np.log10(Z),
        x=X,
        y=Y,
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
    layout_xaxis_range=[np.nanmin(X),np.nanmax(X)],
    layout=fig_layout_extmap)
    return newfig


@sda.callback(
    [Output('extmap-graph', 'figure'),
     Output('extmap-msg', 'children'),
     Output('extmapsignal', 'data'),
    ],
    [Input('extmap-submit-button', 'n_clicks'),
    ],
    [State('extmap-lon', 'value'), 
     State('extmap-lat', 'value'),
     State('extmap-dmin', 'value'),
     State('extmap-dmax', 'value'),
     State('extmap-size', 'value'),
     State('extmap-res', 'value'),
     State('extmap-frame', 'value'),
     State('cube-dropdown', 'value'), 
     ], 
    )
def compute_extmap(clicks,lon,lat,dmin,dmax,size,res,frame,cubename):

    # add check to make sure map size is max 10 degrees.
 
    msg = html.Div([html.Label("message placeholder")])
    input_dict = {}
    newfig = fig_extmap
    results = 0

    if ((clicks == 0) or (clicks == None)):
    
    # #if clicks==0 or clicks is None:

        #set defaults for the initial map to be shown. TBD - choose interesting region. e.g. Chamaelon or Upper Scorpius.
        
        input_dict['lon'] = 5.0 # galactic/ICRS (in degrees)
        input_dict['lat'] = 20.0 # galactic/ICRS (in degrees)
        input_dict['frame'] = 'galactic' # coordinate frame
        input_dict['dmin'] = 50.0   # min distance (in pc)
        input_dict['dmax'] = 500.0  # max distance (in pc)
        input_dict['size'] = 30.0     # angular size of the map (in degrees)
        input_dict['res'] = 3.0      # resolution element of the map (in degrees)
        input_dict['cubename'] = cubename

        msg = html.Div([html.Label("assigned default"+str(input_dict)+str(type(input_dict['size'])))])

        X=np.array([-10.,  -7.,  -4.,  -1.,   2.,   5.,   8.,  11.,  14.,  17.])
        Y=np.array([ 5.,  8., 11., 14., 17., 20., 23., 26., 29., 32.])
        Z=np.array([[0.84744, 0.78793, 0.71737, 0.65173, 0.5835 , 0.52066, 0.47516, 0.43102, 0.37547, 0.33204],
        [0.79036, 0.73365, 0.68123, 0.65283, 0.62454, 0.58294, 0.53322, 0.47591, 0.41212, 0.36403],
        [0.803  , 0.76065, 0.69571, 0.67471, 0.67714, 0.67333, 0.62828, 0.54667, 0.46246, 0.40357],
        [0.90348, 0.84479, 0.74858, 0.72336, 0.75192, 0.79331, 0.76051, 0.63901, 0.51907, 0.44545],
        [0.95923, 0.89575, 0.80907, 0.79647, 0.84263, 0.89778, 0.85753, 0.69842, 0.54692, 0.46065],
        [0.93929, 0.89454, 0.86208, 0.8874 , 0.94341, 0.97079, 0.89704, 0.72226, 0.55891, 0.4625 ],
        [0.94633, 0.92914, 0.95571, 1.01989, 1.06285, 1.01996, 0.88939, 0.71474, 0.55774, 0.45197],
        [1.03538, 1.06985, 1.13871, 1.20782, 1.19607, 1.06071, 0.86089, 0.67619, 0.52789, 0.41675],
        [1.28856, 1.38162, 1.46232, 1.47193, 1.35397, 1.11429, 0.83687, 0.62115, 0.47496, 0.36982],
        [1.7053 , 1.79848, 1.80042, 1.66656, 1.42396, 1.11162, 0.7935 , 0.55839, 0.41534, 0.32257]])

        newfig = update_planar(X,Y,Z)

    else:
        # take inputs as provide by users. 
        # check all fields are provided.
        inputs = [lon,lat,dmin,dmax,size,res,frame]

        if all(v is not None for v in inputs):
            msg = html.Div([
                html.Label('Input complete'),
                html.Pre(str(type(inputs))+str(size/res)),
            ])
        else:
            msg = html.Div([
                html.Label('Missing input'),
                html.Pre(str(type(inputs))),
            ])
            return [newfig, msg, results]

        if all(v is not None for v in inputs):
            ## check size
            if (size/res >= 1000.0):
                msg = html.Div([
                    html.Label('Requested map size (size/resolution:'+str(np.round(size/res,3))+') is too large; Please make sure it is <= 1000')
                    ])
                return [newfig, msg, results]
            
            else:
                try:
                    ### create input_dict
                    input_dict['lon'] = lon
                    input_dict['lat'] = lat
                    input_dict['frame'] = frame
                    input_dict['dmin'] = dmin
                    input_dict['dmax'] = dmax
                    input_dict['size'] = size
                    input_dict['res'] = res
                    input_dict['cubename'] = cubename

                    msg = html.Div([
                        html.Label('assigned input_dict:'),
                        html.Pre(str(input_dict)),
                        html.Pre(str(type(lon)))
                    ])

                except:
                    msg = html.Div([
                        html.Label('Error initiating input_dict'),
                        html.Pre(str(inputs))
                    ])

        input_dict2 = json.dumps(input_dict)

        try: 
            ### run the compute 
            results = global_store(input_dict2)
            X=results['X']
            Y=results['Y']
            Z=results['Z']
            newfig = update_planar(X,Y,Z)
            msg = html.Div([
                html.Label('Map computed'),
                html.Pre("input_dic2:"+str(input_dict2)),
                html.Pre("results:"+str(results)),
                ])

        except:
            msg = html.Div([
                    html.Label('Error - No ouput created'),
                    #html.Pre(str(input_dict2)),
                    #html.Pre(str(results)),
                ])

    # parsed_input = html.Div([
    #     html.Label('status:'),
    #     html.Div([html.Pre(err_msg)])
    # ])

    return [newfig, msg, results]


### function to perform the compution of the integrated extinction sky map.
@cache.memoize()
def global_store(input_dict):
    import numpy as np

    err_message = ''
    results = {}

    input_dict = json.loads(input_dict)

    try:
        lon=input_dict['lon']
        lat=input_dict['lat']
        frame=input_dict['frame']
        dmin=input_dict['dmin']
        dmax=input_dict['dmax']
        size=input_dict['size']
        res=input_dict['res']
        cubename = input_dict['cubename']

        # create grid of coordinates
        lon_grid = np.arange(lon-size/2,lon+size/2,res)
        lat_grid = np.arange(lat-size/2,lat+size/2,res)
        extmap = np.zeros([len(lon_grid),len(lat_grid)])

        results['X']=np.array([1,2,3])
        results['Y']=np.array([4,5,6])
        results['Z']=np.array([[1,1,1],[5,5,5],[9,9,9]])
        results['lon'] = lon
        results['lat'] = lat
        results['frame'] = frame
        results['dmin'] = dmin
        results['dmax'] = dmax
        results['size'] = size
        results['res'] = res
        results['cubename'] = cubename

        #open correct cube
        if cubename == 'cube1':
            cube=globals.cube25
            axes=globals.axes25
            max_axes=globals.max_axes25
        if cubename == 'cube2':
            cube=globals.cube50
            axes=globals.axes50
            max_axes=globals.max_axes50
        if cubename == 'cube1_v2':
            cube=globals.cube_v2_10
            axes=globals.axes_v2_10
            max_axes=globals.max_axes_v2_10
        if cubename == 'cube2_v2':
            cube=globals.cube_v2_25
            axes=globals.axes_v2_25
            max_axes=globals.max_axes_v2_25
        if cubename == 'cube3_v2':        
            cube=globals.cube_v2_50
            axes=globals.axes_v2_50
            max_axes=globals.max_axes_v2_50
        
        results['maxaxes'] = max_axes

        try:
            for i in range(len(lon_grid)):
                for j in range(len(lat_grid)):
                    sc=SkyCoord(lon_grid[i]*u.deg, lat_grid[j]*u.deg, frame=frame)
                    bins, ext = reddening_distance_bin(sc, d_bins=[dmin,dmax], cube=cube, axes=axes, max_axes=max_axes, step_pc=5)
                    extmap[i,j] = ext

            results['status'] = 'extmap ok'
                #sc=SkyCoord(lon_grid[0]*u.deg, lat_grid[0]*u.deg, frame=frame)
                #bins, ext = reddening_distance_bin(sc, d_bins=[dmin,dmax], cube=cube, axes=axes, max_axes=max_axes, step_pc=5)
                #results['sc'] = str(sc)
                #results['bins']=bins
                #results['ext']=ext
        except:
            results['bins']='no bins'
            results['ext']='no ext'

        results['X']=lon_grid
        results['Y']=lat_grid
        results['Z']=extmap


    except:
        results['X']=np.array([0,5,10])
        results['Y']=np.array([0,5,10])
        results['Z']=np.array([[0,0,0],[0,0],[0,0,0]])
        results['lon'] = 0
        results['lat'] = 0
        results['frame'] = 'failed'
        results['dmin'] = 0
        results['dmax'] = 0
        results['size'] = 0
        results['res'] = 0
        results['cubename'] = 'failed'

    return results



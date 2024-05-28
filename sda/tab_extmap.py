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
import os
import h5py
from numpy import errstate,isneginf,array


from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

import globals

from sda import sda

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

""" setup the slider range """
# Now also computing skymaps for a 5-pc sampled d-grid from 0-5000 pc,
# but this is too fine grained for the slider.
# we can keep the slider below or reduce the number of points. 
# For compatibility keep d-grid with multples of 5.

range_d=np.concatenate(( np.linspace(0,1000,101), np.linspace(1025,1500,20), np.linspace(1550,3000,30), np.linspace(3100,5000,20) ))
range_d = range_d[1:]

range_slider_marks = {}

for d in range(len(range_d)):
        range_slider_marks[np.log(range_d[d])] = '' #{'label':'X', 'style': {'color':'#f50'} }

range_slider_marks[2.30258509] = "10 pc"
range_slider_marks[3.91202301] = "50 pc"
range_slider_marks[4.60517019] = "100 pc"
range_slider_marks[6.2146081]  = "500 pc"
range_slider_marks[6.90775528] = "1000 pc"
range_slider_marks[8.51719319] = "5000 pc"

# range_slider_marks = {}
# i=0
# for d in range(len(range_d)):
#     if (i == 10): i=0
#     if i == 9:
#         range_slider_marks[int(range_d[d])] = str(int(range_d[d]))+"pc"
#     else:
#         range_slider_marks[int(range_d[d])] = '' #{'label':'X', 'style': {'color':'#f50'} }
#     i=i+1
     
# i=0
# marks = {}
# for d in range(len(range_d)):
#     dkey=int(range_d[d])
#     if (i==10): i=0
#     if i == 9:
#         marks[dkey] = str(dkey)+"pc"     
#     else:
#          marks[dkey] = ''
#     i=i+1
# marks[3000.50] = '3000.50'

# i=0
# marks = {}
# for d in range(len(range_d)):
#     dkey=int(range_d[d])
#     marks[dkey] = ''
#     i=i+1
# marks[0] = '0pc'
# marks[200] = '200pc'
# marks[500] = '500pc'
# marks[1000] = '1000pc'
# marks[2000] = '2000pc'
# marks[3000] = '3000pc'
# marks[4000] = '4000pc'
# marks[5000] = '5000pc'

""" setup a test lbd-ext-grid """
def create_test():
    lgrid = np.linspace(-np.pi, np.pi, 721)
    bgrid = np.linspace(-np.pi/2, np.pi/2, 361)
    dgrid = np.concatenate(( np.linspace(0,1000,101), np.linspace(1025,1500,20), np.linspace(1550,3000,30), np.linspace(3100,5000,20) ))
    extdata = np.random.rand(len(lgrid),len(bgrid),len(dgrid))
    h5f = h5py.File('testgrid.h5', 'w')
    h5f.create_dataset('l', data=lgrid, compression='gzip', compression_opts=9) #dtype defaults to 'f'
    h5f.create_dataset('b', data=bgrid, compression='gzip', compression_opts=9)
    h5f.create_dataset('d', data=dgrid, compression='gzip', compression_opts=9)
    h5f.create_dataset('e', data=extdata, compression='gzip', compression_opts=9)
    h5f.close()
    print(extdata)
    print(extdata.shape)
    return

def content_extmap():
    content_extmap = html.Div([
        #dcc.Store(id='memory'),
        html.Br(),
        html.Div([
            html.H5("Extinction Sky Map"),
            html.Pre("Move the slider min-max to select the extinction integration range"),
            html.Div([
                dcc.RangeSlider(
                    id='extmap-slider',
                    min=2.3,
                    max=8.52,
                    step=None,
                    value=[2.30258509,6.90775528], # 10 to 1000 pc (ln)
                    allowCross=False,
                    marks = range_slider_marks,
                    #updatemode='drag',
                ),
            ], id='dslider'),
            html.Div(id='extmap-msg', children=[html.Label('Selected range (pc)')]),
            #], style={'width':'100%'}),
            dcc.Graph(id='extmap-graph', figure=go.Figure(), config={"doubleClick": "reset"}),
            html.Div([  
                html.Button(id='save-extmap', n_clicks=0, children='Export map', style={'margin-left':'10px'}),
                dcc.Download(id='download-extmap-csv'),
            ]),
            html.Br(),
            html.Div([
                html.Label('Z-scale:'),
                dcc.RadioItems(
                    id='linlog',
                    options=['Log', 'Linear'],
                    value='Log',
                ),
                html.Div(
                    id='z',
                    children=[
                        dcc.RangeSlider(
                        id='zscale',
                            min=-3.0,
                            max=3.0,
                            value=[-1.5,1.0],
                            allowCross=False,
                            marks={
                                -3: {'label':'-3'},
                                -2: {'label':'-2'},
                                -1: {'label':'-1'},
                                0: {'label':'0'},
                                1: {'label':'1'},
                                2: {'label':'2'},
                                3: {'label':'3'},
                            },
                            included=True,
                        ),
                    ],
                ),
            ], style={'width':'50%'}),
            dcc.Markdown(''' 
                Info: Skymap reconstructed from V2 cubes (10 pc resolution up to 1500 pc, 25 pc resolution from 1500 to 3000 pc and 50 pc resolution from 3000 to 5000 pc).
                '''),
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

    ], id='extmap', style={'width':'100%', 'display':'inline-block', 'margin':'auto', 'padding-left':'10%', 'padding-right':'10%'})

    return content_extmap


""" functions """

fig_layout_extmap = dict(
        margin={'l': 5, 'b': 5, 't': 5, 'r': 5},
        legend={'x': 0.8, 'y': 0.1},
        hovermode='closest',
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        yaxis=dict(
            scaleanchor='x',
            scaleratio=1,
            #automargin=True,
        ),
        #xaxis=dict(
        #    automargin=True,
        #),
        #uirevision='constant',
        #autosize=False,
        #minreducedwidth=400,
        #minreducedheight=200,
    )

def update_extheat(X,Y,Z,zscale,linlog):
    if linlog == "Log":
        text = '(log10)'
    else:
        text = ''
    newfig = go.Figure(data=
    go.Heatmap(
        z=Z.T,
        x=X,
        y=Y,
        name='Integrated extinction',
        colorscale='Cividis_r', #'Inferno_r', #
        colorbar=dict(
            title='A(550nm) mag '+text,
            titleside="right",
            ),
        type = 'heatmap',
        zmin = zscale[0],
        zmax = zscale[1],
        #connectgaps=True, 
        #zsmooth='fast',
        hoverinfo='all',
        hovertemplate="l:%{x:.1f}, b: %{y:.1f}, ext "+text+": %{z:.5f}",
    ),
    layout=fig_layout_extmap
    )    
    return newfig


""" callbacks """
@sda.callback(
    [
     Output('z', 'children'),
    ],
    [
     Input('linlog', 'value'),     
    ]
)
def update_slider(linlog):
    if linlog == 'Log':
        newslider = dcc.RangeSlider(
            id='zscale',
            min=-3.0,
            max=3.0,
            value=[-1.5,1.0],
            allowCross=False,
            marks={
                -3: {'label':'-3'},
                -2: {'label':'-2'},
                -1: {'label':'-1'},
                0: {'label':'0'},
                1: {'label':'1'},
                2: {'label':'2'},
                3: {'label':'3'},
            },
            included=True,
        )
    else: 
        newslider = dcc.RangeSlider(
            id='zscale',
            min=0.0,
            max=7.0,
            value=[0.1,2.0],
            allowCross=False,
            marks={
                0: {'label':'0'},
                1: {'label':'1'},
                2: {'label':'2'},
                3: {'label':'3'},
                4: {'label':'4'},
                5: {'label':'5'},
                6: {'label':'6'},
                7: {'label':'7'},
            },
            included=True,
        )

    return [newslider]


@sda.callback(
    [
     Output('dslider', 'children'),
    ],
    [
     Input('cube-dropdown', 'value'),     
    ]
)
def update_dslider(cubename):

    if (cubename == 'cube1') or (cubename == 'cube2'):
        # this service is only for V2 cubes. no slider.
        return [html.Div(id='extmap-slider')]        
    
    if (cubename == 'cube1_v2'):
            min=2.30
            max=7.32

    if (cubename == 'cube2_v2'):
            min=2.30
            max=8.01

    if (cubename == 'cube3_v2'):
            min=2.30
            max=8.52

    dslider = dcc.RangeSlider(
                id='extmap-slider',
                min=min,
                max=max,
                step=None,
                value=[2.30258509,6.90775528], # 10 to 1000 pc (ln)
                allowCross=False,
                marks = range_slider_marks,
                #updatemode='drag',
            ),

    return [dslider]    

@sda.callback(
    [Output('extmap-graph', 'figure'),
     Output('extmap-msg', 'children'),
    ],
    [Input('extmap-slider', 'value'),
     Input('zscale', 'value'),
     Input('linlog', 'value'),
    ],
    [
    State('cube-dropdown', 'value'),
    ]
    )
def compute_extmap(value,zscale,linlog,cubename):
    import h5py

    msg = html.Div([html.Label("Selected range (pc) -- d(min): "+str(np.round(np.exp(value[0])))+"; d(max):"+str(np.round(np.exp(value[1]))))])
    
    newfig = go.Figure()

    if (cubename == 'cube1') or (cubename == 'cube2'):
        # this service is only for V2 cubes
        msg=html.Div([html.Label('Please select one of the three V2 cubes')])
        return [newfig, msg]        
    
    # update file names once created.
    
    if (cubename == 'cube1_v2'):
            grid_data = "grid_lbdext_0.5deg_5pc.h5"
    if (cubename == 'cube2_v2'):
            grid_data = "grid_lbdext_0.5deg_5pc.h5"
    if (cubename == 'cube3_v2'):
            grid_data = "grid_lbdext_0.5deg_5pc.h5"
    
    input_h5 = os.path.join(os.environ.get('SERVICE_APP_DATA'), grid_data)    
    h5f = h5py.File(input_h5,'r')

    l = np.rad2deg(h5f['l'][:])
    b = np.rad2deg(h5f['b'][:])
    d = h5f['d'][:]

    idx1 = np.where( d == np.round(np.exp(value[0]),5) )
    idx2 = np.where( d == np.round(np.exp(value[1]),5) )

    ext1=h5f['e'][:,:,idx1[0][0]]
    ext2=h5f['e'][:,:,idx2[0][0]]

    diffext=ext2-ext1

    if linlog == "Log":
        #extmap = np.log10(diffext, out=np.zeros_like(diffext), where=(diffext!=0))
        #extmap = np.log10( ext2 - ext1 + 1.0e-7 )

        with errstate(divide='ignore'):
            extmap = np.log10(diffext)
            extmap[isneginf(extmap)]=0
            ##

    else: 
        extmap = diffext

    newfig = update_extheat(l,b,extmap,zscale,linlog)

    h5f.close()

    # X = np.linspace(-180.0,180.0,361)
    # Y = np.linspace(-90,90,181)
    # d = np.linspace(0,5000,171)
    # ZZ=np.random.rand(len(X),len(Y),len(d))
    # Z=ZZ[:,:,50]-ZZ[:,:,20]
   
    msg = html.Div([
        #html.Pre('l:'+str(l.shape)),
        #html.Pre('b:'+str(b.shape)),
        #html.Pre('d:'+str(d.shape)),
        #html.Pre('idx1:'+str(idx1[0][0])),
        #html.Pre('idx2:'+str(idx2[0][0])),
        #html.Pre('d2:'+str(d2.shape)),
        #html.Pre('ext:'+str(ext1.shape)),
        #html.Pre('ext:'+str(ext2.shape)),
        #html.Pre('extmap:'+str(extmap[:,:].shape)),
        #html.Pre('newextmap:'+str(newextmap.shape)),
        #html.Pre('X:'+str(X.shape)),
        #html.Pre('Y:'+str(Y.shape)),
        #html.Pre('Z:'+str(Z.shape)),
        #html.Pre('ZZ:'+str(ZZ.shape)),
        #html.Pre('zscale:'+str(zscale)),
        html.Pre("selected range: d(min)="+str(np.round(np.exp(value[0])))+"; d(max)"+str(np.round(np.exp(value[1])))),
        ])
   
    return [newfig, msg]


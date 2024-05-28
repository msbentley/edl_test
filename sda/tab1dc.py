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

import io
import base64
import datetime
import pandas as pd
import json

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
#from astroquery.gaia import Gaia

import scipy.interpolate as spi

import dash
#import dash_html_components as html
#import dash_core_components as dcc
#import dash_table
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go

import globals

from sda import sda
from reddening import reddening
from reddening_distance import reddening_distance

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


def content_1dc():
    content_1dc = html.Div([
        html.Div([
            html.Br(),
            html.H5("Upload Target List"),
            html.Div([
                html.Label('Input list (l,b,d)'),
                dcc.Upload(
                        id='upload-file2',
                        children=[html.Button(['Upload file'])],
                        multiple=False
                ),
                html.Br(),
                dcc.RadioItems(
                        id='check-lb2',
                        options=[
                            {'label': 'Galactic coordinates (l,b)', 'value': "1"},
                            {'label': 'Equatorial coordinates (ra,dec)', 'value': "0"},
                        ],
                        value = "1",
                ),
                html.Br(),
                html.Button(id='submit-button-1d-ext', n_clicks=0, children='Submit'),
                html.H6('Check parsing input file:'),
                html.Div(id='uploaded-targets2', style={'width':'100%', 'padding':'10px'}),
                html.Div(id='set-ra'),
                html.Div(id='set-dec'),
            ], style={'padding':'3px', 'margin':'10px', 'borderWidth': '2px'}),
            dcc.Loading(
                id='load-1dc',
                type='circle',
                fullscreen=False,
                children=[
                    html.Div(
                            id='tab1dc-content', 
                            children=[],
                    ),
                ],
            ),
        ], style={'width': '25%', 'float':'left', 'display':'inline-block', 'margin-right':'5px'}),
        html.Div([
            html.Br(),
            html.H5('Results'),
            #html.Br(),
            html.Div(id='1dc-status'),
            html.Br(),
            html.Div([   #add download button to save results integrated extinction
                html.Button(id='save-gaia-1dc', n_clicks=0, children='Download CSV', style={'margin-left':'0px'}),
                dcc.Download(id='download-intext-csv'), 
            ]),
            #html.Br(),
            #html.Div(id='dropdown-div'),
            #html.Div(id='log'),
            html.Div(id='intext-result'),
        ], style={'width': '50%','float':'left','display':'inline-block', 'margin-right':'5px'})
    ])

    return content_1dc


######### CALLBACKS ########


### function to perform the compution of integrated extinction for number of sightlines.
@cache.memoize()
def global_store(input_dict):
    import numpy as np

    input_dict = json.loads(input_dict)
    step_pc=input_dict['step_pc']
    ra=input_dict['ra']
    dec=input_dict['dec']

    sc=SkyCoord(ra*u.deg, dec*u.deg)
    target_dist = input_dict['dist']

    cubename = input_dict['cubename']

    dist=None
    ext=None
    exterr=None
    cube_id=None

    err_message = ''

    results = {}

    out_ra = []
    out_dec = []
    out_dist = []
    out_ext = []
    out_exterr = []

    try:
        for i in range(len(sc)):
                
            if (cubename == 'cube1') or (cubename == 'cube2'):
                try:                
                    # extract the error on the intergrated extinction; used for the cumulative extinction plot
                    dist, exterr = reddening_distance(sc[i], target_distance=target_dist[i], cube=globals.cube_errext, axes=globals.axes_errext, max_axes=globals.max_axes_errext,step_pc=step_pc,error=True)
                except:
                    err_message = 'error read error cubes'+str(sc[i].galactic)

            if (cubename == 'cube1_v2') or (cubename == 'cube2_v2') or (cubename == 'cube3_v2'):
                try:                
                    # extract the error on the intergrated extinction; used for the cumulative extinction plot
                    dist, exterr = reddening_distance(sc[i], target_distance=target_dist[i], cube=globals.cube_errext2, axes=globals.axes_errext2, max_axes=globals.max_axes_errext2,step_pc=step_pc,error=True)
                except:
                    err_message = 'error read error cubes'+str(sc[i].galactic)

            if cubename == 'cube1':
                cube_id = "explore_cube_density_values_025pc_v1.h5"
                dist, ext = reddening_distance(sc[i], target_distance=target_dist[i], cube=globals.cube25, axes=globals.axes25, max_axes=globals.max_axes25,step_pc=step_pc)

            if cubename == 'cube2':
                cube_id = "explore_cube_density_values_050pc_v1.h5"
                dist, ext = reddening_distance(sc[i], target_distance=target_dist[i], cube=globals.cube50, axes=globals.axes50, max_axes=globals.max_axes50,step_pc=step_pc)

            if cubename == 'cube1_v2':
                cube_id = "explore_cube_density_values_010pc_v2.h5"
                dist, ext = reddening_distance(sc[i], target_distance=target_dist[i], cube=globals.cube_v2_10, axes=globals.axes_v2_10, max_axes=globals.max_axes_v2_10,step_pc=step_pc)

            if cubename == 'cube2_v2':
                cube_id = "explore_cube_density_values_025pc_v2.h5"
                dist, ext = reddening_distance(sc[i], target_distance=target_dist[i], cube=globals.cube_v2_25, axes=globals.axes_v2_25, max_axes=globals.max_axes_v2_25,step_pc=step_pc)

            if cubename == 'cube3_v2':
                cube_id = "explore_cube_density_values_050pc_v2.h5"
                dist, ext = reddening_distance(sc[i], target_distance=target_dist[i], cube=globals.cube_v2_50, axes=globals.axes_v2_50, max_axes=globals.max_axes_v2_50,step_pc=step_pc)

            err_message=str(cube_id)

            out_ra.append(ra[i])
            out_dec.append(dec[i])
            out_dist.append(dist)
            out_ext.append(ext)
            out_exterr.append(exterr)

        #results['dataset'] = str(cube_id)
        results['ra'] = out_ra
        results['dec'] = out_dec
        results['dist'] = out_dist
        results['ext'] = out_ext
        results['exterr'] = out_exterr
        #results['err_msg'] = err_message

    except:

        #results['dataset'] = str(cube_id)
        results['ra'] = 0
        results['dec'] = 0
        results['dist'] = 0
        results['ext'] = 0
        results['exterr'] = 0
        results['err_msg'] = err_message
 
    return results

### callback to parse the list of input targets and initiate the computation of integrated extinctions.
@sda.callback(
    [Output('extsignal', 'data'),
     Output('1dc-status', 'children'),
     Output('intext-result', 'children'),
    ],
    [Input('submit-button-1d-ext', 'n_clicks'),
    ],
    [State('check-lb2', 'value'),
    State('upload-file2', 'contents'),
    State('upload-file2', 'filename'),
    State('cube-dropdown', 'value'),
    State("extsignal", "data"),
    ],
)
def compute_value(n_clicks1,checklb,csv_content,csv_filename,cubename,extsignal_data):
    # compute value and send a signal when done
    err_message=''
    sc=None
    input_dict = {}
    input_dict2 = {}
    results = None

    if ((n_clicks1 == 0) or (n_clicks1 == None)):
        raise PreventUpdate

    ctx = dash.callback_context

    if not ctx.triggered: 
        button_id = None
        raise PreventUpdate

    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    import json
    ctx_msg = json.dumps({
        'states': ctx.states,
        'triggered': ctx.triggered,
        'inputs': ctx.inputs,
        #'outputs': ctx.outputs_list,
    }, indent=2)

    if (button_id == 'submit-button-1d-ext'):
        lon=None
        lat=None

        if csv_filename is not None:
            err_message = '' #str(csv_filename)
            if 'csv' in csv_filename:
                try:
                    content_type, content_string = csv_content.split(',')
                    decoded = base64.b64decode(content_string)
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

                    if len(df.columns) !=3:
                        return

                    elif len(df.columns) == 3:
                        #try to use as coordinates: lon, lat
                        df_lon = df.iloc[:,0]
                        df_lat = df.iloc[:,1]
                        dista = df.iloc[:,2]
                        resolve = 0
                        lon=df_lon.to_numpy()
                        lat=df_lat.to_numpy()
                        dist = dista.to_numpy()
                        #err_message = str(lon)
                        targets = [None]

                    else:
                        err_message = 'warning: no valid csv file detected; please check the required format'
                
                except:
                    err_message = 'warning: no valid csv file detected; parsing failed'
        else:
            err_message = "please first upload target list"

    if ((lon is not None) and (lat is not None)):

        if checklb == "1":
            sc=SkyCoord(lon*u.deg, lat*u.deg, frame='galactic')
            sc=sc.transform_to('icrs')

            err_message = "Galactic coordinate converted to ICRS: "+str(sc)

        if checklb == "0": 
            sc=SkyCoord(lon*u.deg, lat*u.deg, frame='icrs') 

            err_message = "Used ICRS coordiantes: "+str(sc)

    else:
        err_message = "Please provide valid lon/lat values (not None)"
            
    if not sc:

        err_message2 = 'No SkyCoord provided'  

    else:
        #coord_string = "used coordinates: "+str(sc.ra)+"_"+str(sc.dec)
        #x_err_message = "run global_store()"

        err_message = str(sc.ra.value)

        try:
            input_dict['ra'] = (sc.ra.value).tolist()
            input_dict['dec'] = (sc.dec.value).tolist()
            input_dict['dist'] = dist.tolist()
            input_dict['step_pc'] = 5
            input_dict['index'] = np.arange(len(sc)).tolist()
            input_dict['cubename'] = cubename

            input_dict2 = json.dumps(input_dict)

            err_message = "Assigned valued to input_dict"+str(input_dict2)


        except:
            err_message = "Issue with input_dict"

        try:
            results = global_store(input_dict2) ## this is the expensive computation to compute integrated extinctions.
            err_message = "send to global_storage"

        except:
            err_message = "error with global store calculation"

    #if sc is not None:
    #    #options=[{'label': str(sc[i].ra)+"_"+str(sc[i].dec), 'value': str(sc[i].ra)+"_"+str(sc[i].dec)} for i in range(len(sc))]
    #    droplist = dcc.Dropdown(
    #        id='1d-dropdown',
    #        options=[{'label': str(i)+"_"+str(sc[i].ra)+"_"+str(sc[i].dec), 'value': str(i)+"_"+str(sc[i].ra)+"_"+str(sc[i].dec)} for i in range(len(sc))],
    #        clearable=True,
    #    )

    try:
        df = pd.DataFrame()
        #results = global_store(extsignal_data)
        df = pd.DataFrame.from_dict(results, orient='index')
        df = df.transpose()

        intext_table = dash_table.DataTable(
            id='table_intext',
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i } for i in df.columns],
            #filter_action='native',
            #editable=True,
            #row_selectable='multi',
            #row_deletable=True,
            #selected_rows=[],
            export_format='csv',
            export_headers='names',
            page_action='native',
            page_current=0,
            page_size=10,
            fixed_rows={'headers': True},
            style_table={'height': '300px', 'overlfowY': 'auto'}
        ),
    
        err_message = 'success intext'

    except:
        inteext_table = html.Pre("failed intext")
        err_message = 'failed intext'

    div_status=html.Div([
        #html.Pre("status info!"),
        #html.Pre(str(button_id)),
        #html.Pre(str(input_dict)),
        #html.Pre(str(len(input_dict['ra']))),
        #html.Pre(err_message),
    ])

    return [results, div_status, intext_table]


### callback to parse the uploaded csv - allows user to verify the input is read correctly.
@sda.callback(Output('uploaded-targets2', 'children'),
              Input('upload-file2', 'contents'),
              State('upload-file2', 'filename'),
              State('upload-file2', 'last_modified') )
def update_targets(content, filename, moddate):
    if content is not None:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                test=[{'name': i, 'id': i } for i in df.columns]
                #add check to make sure it's a 3-col input with coordiantes + distance
        except:
            return html.Div(['Error in processing this file; please ensure to provide csv file formatted (l,b,d) or (ra,dec,d)'])

        return html.Div([
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(moddate)),
            dash_table.DataTable(
                id='datatable-bulk',
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i } for i in df.columns],
                #filter_action='native',
                #editable=True,
                #row_selectable='multi',
                #row_deletable=True,
                #selected_rows=[],
                page_action='native',
                page_current=0,
                page_size=10,
                fixed_rows={'headers': True},
                style_table={'height': '300px', 'overlfowY': 'auto'}
            ),
            html.Hr(),
        ], style={'width':'90%', 'padding':'5px'}) 
 

### callback to save 1d integrated extinctions. Format of output file TO BE CONFIRMED.
@sda.callback(
    [Output("download-intext-csv", "data"),
     #Output('log', 'children')
     ],
    [Input("save-gaia-1dc", "n_clicks"),
     Input("extsignal", "data")
     ],
    [State('cube-dropdown', 'value')],
    prevent_initial_call=True,
)
def save1d(n_clicks, extsignal_data, cubename):

    if n_clicks==0:
        print('update prevented')
        raise PreventUpdate

    else:
        mydict = {}
        err_message = 'no error'
        df = pd.DataFrame()

        try:
            results = extsignal_data
            df = pd.DataFrame.from_dict(results, orient='index')
            df = df.transpose()

        except:
            results = None
            err_msg = 'error on reading store'

        #data = np.column_stack((np.arange(10), np.arange(10) * 2))
        #df = pd.DataFrame(columns=["a column", "another column"], data=data)

    err_msg = "clicks: "+str(n_clicks)


    return [
        dcc.send_data_frame(df.to_csv, filename="intext.csv", index=False)#,
        #html.Div([html.Pre(str(extsignal_data))])
        ]


# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 18:21:02 2019

@author: creze
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.figure_factory as ff
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import json
from textwrap import dedent as d
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import base64
from io import BytesIO



app = dash.Dash(__name__)

df = pd.read_csv('rtr-data.csv')

# maskfile for wordcloud mask
modelmask = np.array(Image.open('mask.png'))
modelmask[modelmask==1]=255

opcolors = {'A1': 'rgb(218,41,28)',
            'H3A': 'rgb(0,0,0)',
            'TMA': 'rgb(226,0,116)'}

#some colors
plot_background = '#f4efef'
paper_background = '#f4efef'
cmap = 'plasma'


app.layout= html.Div(children=
    [
    html.Div([
        #header div
        html.Div([
            dcc.Markdown(d('''
            # Network Benchmark Austria
             ''')),

        ], className='twelve columns'),
    ], className='row'),
    html.Div([
        html.Div([
            dcc.Markdown(d('''
                > Benchmark of networks in Austria, operated by A1, Deutsche Telekom and Hutchison Drei Austria, based on open data from [RTR-Netztest](https://www.netztest.at/de/Opendata)  
                > Only observations, with location info and recorded MNC are used for this dashboard.  
                > Select the __political district__ of interest, and __down- upload-speed__ from dropdown menue (or select area on map) to explore data.
                '''))
        ], className='eight columns'),
        html.Div([
            dcc.Markdown(d('''
            heres the table with some general data. Start & End-Date, Entries, etc.
             ''')),
        ], className='four columns'),
    ], className='row'),



     html.Div([
        #steering element row
        html.Div([
        dcc.Dropdown(id = 'district_dd', options=[
                {'label': i, 'value': i} for i in df.district.sort_values().unique()
                ], value='Korneuburg'),
        ], className='three columns'),
        html.Div([
        dcc.Dropdown(id = 'KPI_selection', options=[
                {'label': 'upload speed', 'value': 'upload_Mbit'},
                {'label': 'download speed', 'value': 'download_Mbit'},
                ], value='download_Mbit',
                    searchable=False,

                     ),
        ], className='three columns'),

     ], className='row',
     style={
         'margin-bottom': 50,
     }),
     html.Div([
        #main graphs row
        html.Div([
            dcc.Graph( id = 'map'),
            dcc.Markdown("In case the selection doesn't update properly, select empty area on map")
        ], className='four columns'),
        html.Div([
            dcc.Graph(id='boxplot_download'),
        ], className='four columns'),
         html.Div([
             dcc.Graph(id='distplot'),
         ], className='four columns'),

     ], className='row'),
    html.Div([
        html.Div([
            dcc.Graph( id = 'observation_count'),
            ], className='four columns'),
            html.Div([
                dcc.Markdown(d('''
                    Used Devices
                    ''')),
                html.Img(id='wordcloud', style={
                    'max-width': '100%',
                }),
        ], className='four columns', style={'text-align': 'center'}),
        html.Div([
            dcc.Graph(id = 'networkmodes'),
                    html.Div([
                        dcc.Markdown(d("""
                        **Selection Data**

                        Choose the lasso or rectangle tool in the graph's menu
                        bar and then select points in the graph.

                        Note that if `layout.clickmode = 'event+select'`, selection data also 
                        accumulates (or un-accumulates) selected data if you hold down the shift
                        button while clicking.
                    """)),
                        html.Div(id='selectedrange')
                    ], style={'display': 'none'}),

                ], className='four columns'),
    ], className='row')



], style={
    'margin-left': '2%',
    'margin-right': '2%',
    #'margin-top': '2%',

    }

)

# helper functions


def selection_is_in(dfseries, selection):
    # check if the selection is in the series of test ids
    # if not all of them are contained return False
    selection = pd.Series(selection)
    try:
        if True in selection.isin(dfseries).values:
            contained = True
        else:
            contained = False
        return contained
    except:
        print('could not check')

@app.callback(
    Output('selectedrange', 'children'),
    [Input('map', 'selectedData')])
def display_selected_data(selectedData):
    t = json.loads(json.dumps(selectedData, indent=2))
    selection = []
    try:
        for p in t['points']:
            selection.append(p['text'])
    except:
        selection = []

    return selection


@app.callback(
        Output(component_id='wordcloud', component_property = 'src'),
        [Input(component_id='district_dd', component_property = 'value'),
         Input('selectedrange', 'children')]
        )
def generate_wordcloud(district_dd, selectedrange):

    dff = df[df.district == district_dd]

    # check if there is points in dff in selected range - if yes, use for selection
    if selectedrange:
        if selection_is_in(dff['open_test_uuid'], selectedrange):
            dff = dff[dff['open_test_uuid'].isin(selectedrange)]

    text = dff.model.to_string()
    wordcloud = WordCloud(width=300, height=200, max_words=50, background_color= plot_background,
                          mask=modelmask, colormap=cmap).generate_from_text(text)

    in_mem_file = BytesIO()
    img = wordcloud.to_image()
    img.save(in_mem_file, format="PNG")
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()

    encoded_image = base64.b64encode(img_bytes)
    imgsrc = 'data:image/png;base64,{}'.format(encoded_image.decode('ascii'))

    return imgsrc


@app.callback(
        Output(component_id='observation_count', component_property = 'figure'),
        [Input(component_id='district_dd', component_property = 'value'),
         Input('selectedrange', 'children')]
        )
def update_observation_count(district_dd, selectedrange):
    dff = df[df.district == district_dd]

    # check if there is points in dff in selected range - if yes, use for selection
    if selectedrange:
        if selection_is_in(dff['open_test_uuid'], selectedrange):
            dff = dff[dff['open_test_uuid'].isin(selectedrange)]

    values = [dff['network'][dff.network == i].count() for i in dff.network.unique()]
    labels = [i for i in dff.network.unique()]
    colors = [opcolors[i] for i in dff.network.unique()]

    trace = go.Pie(values=values, labels=labels,
                   hoverinfo='label+percent', textinfo='value',
                   textfont=dict(size=20),
                   marker=dict(colors=colors,
                               line=dict(color='#000000', width=2)))

    layout = go.Layout(title =  'Count of Observations', legend=dict(orientation='h'), plot_bgcolor=plot_background,
                         paper_bgcolor=paper_background, margin=go.layout.Margin(t=30))

    fig = go.Figure([trace], layout)

    return fig


@app.callback(
        Output('boxplot_download', 'figure'),
        [Input('district_dd','value'),
         Input('KPI_selection', 'value'),
         Input('selectedrange', 'children')
         ])
def update_boxplot(district_dd, KPI_selection, selectedrange):

    dff = df[df.district == district_dd]

    # check if there is points in dff in selected range - if yes, use for selection
    if selectedrange:
        if selection_is_in(dff['open_test_uuid'], selectedrange):
            dff = dff[dff['open_test_uuid'].isin(selectedrange)]

    netsorted = dff.network.sort_values().unique()

    kpi1 = KPI_selection
    data = [go.Box(y = dff[dff.network==i][kpi1], name=i, marker=dict(color = opcolors[i]) ) for i in netsorted]
    layout = go.Layout(showlegend=True, legend=dict(orientation='h'), plot_bgcolor=plot_background, paper_bgcolor=paper_background,
                       margin=go.layout.Margin(t=30))

    fig = go.Figure(data=data, layout=layout)
    fig['layout'].update(title='Boxplot {}'.format(kpi1))
    fig.update({'layout': {'yaxis': {'title': kpi1}}})
    return fig


@app.callback(
        Output('networkmodes', 'figure'),
        [Input('district_dd','value'),
         Input('KPI_selection', 'value'),
         Input('selectedrange', 'children')
         ])
def update_networkmodes(district_dd, KPI_selection, selectedrange):

    dff = df[df.district == district_dd]

    # check if there is points in dff in selected range - if yes, use for selection
    if selectedrange:
        if selection_is_in(dff['open_test_uuid'], selectedrange):
            dff = dff[dff['open_test_uuid'].isin(selectedrange)]

    data = []
    netsorted = dff.network.sort_values().unique()

    for net in dff.network_type.unique():
        trace = go.Bar(x=netsorted,
                       y=[dff[(dff.network_type == net) & (dff.network == i)]['network_type'].count() for i in netsorted],
                       name=net,)
        data.append(trace)

        layout = {
            'yaxis': {'title': 'Network Type'},
            'barmode': 'stack',
            'title': 'Network Types',
        };

    fig = go.Figure(data=data, layout=layout)
    fig['layout'].update(legend=dict(orientation='h'), plot_bgcolor=plot_background,
                         paper_bgcolor=paper_background, margin=go.layout.Margin(t=30))

    return fig


@app.callback(
        Output('distplot', 'figure'),
        [Input('district_dd','value'),
        Input('KPI_selection', 'value'),
        Input('selectedrange', 'children')])
def update_distplot(district_dd, KPI_selection, selectedrange):
    dff = df[df.district == district_dd]

    # check if there is points in dff in selected range - if yes, use for selection
    if selectedrange:
        if selection_is_in(dff['open_test_uuid'], selectedrange):
            dff = dff[dff['open_test_uuid'].isin(selectedrange)]

    netsorted = dff.network.sort_values().unique()

    data = [dff[KPI_selection][dff.network == i] for i in netsorted]
    group_labels = netsorted
    colors = [opcolors[i] for i in netsorted]

    fig = ff.create_distplot(data, group_labels, bin_size=5, show_hist=False, show_rug=False, colors=colors)
    fig['layout'].update(title='Distplot {}'.format(KPI_selection), legend=dict(orientation='h'), plot_bgcolor=plot_background,
                         paper_bgcolor=paper_background, margin=go.layout.Margin(t=30))

    return fig

@app.callback(
        Output('map', 'figure'),
        [Input('district_dd', 'value'),
         Input('KPI_selection', 'value'),
         ])
def update_map(district_dd, KPI_selection):

    dff = df[df.district == district_dd]
    # mapbox_access_token = 'ADD_YOUR_TOKEN_HERE'
    mapbox_access_token = "pk.eyJ1Ijoia3Jpczk5IiwiYSI6ImNqcXZjdThjdDB0enE0OG9iYzJyOTVra2QifQ.yrkexLgmrtCMo9FRxeGiwg"

    netsorted = dff.network.sort_values().unique()

    data = [go.Scattermapbox(
        lat=dff['lat'][dff.network == i],
        lon=dff['long'][dff.network == i],
        mode='markers',
        marker=dict(size=5, opacity=0.5, color=opcolors[i]),
        text=dff['open_test_uuid'],
        name=i
    ) for i in netsorted]

    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            style='outdoors',
            #style='dark',
            #style='light',
            center=dict(
                lat=dff.lat.mean(),
                lon=dff.long.mean()
            ),
            pitch=0,
            zoom=10
        ),
        margin=dict(
            l=0,
            r=0,
            t=30,
            ),
        legend=dict(orientation='h'),
        paper_bgcolor=paper_background,
        plot_bgcolor=plot_background,
        title=district_dd,
    )
    fig = dict(data=data, layout=layout)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

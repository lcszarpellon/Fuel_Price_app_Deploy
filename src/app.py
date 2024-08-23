import pandas as pd
import dash
from dash import html,dcc,Input,Output,State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_templates import ThemeSwitchAIO

# ========= App ============== #
FONT_AWESOME = ["https://use.fontawesome.com/releases/v5.10.2/css/all.css"]
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.4/dbc.min.css"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc_css])
app.scripts.config.serve_locally = True
server = app.server


# ========== Styles ============ #

template_theme1 = "minty"
template_theme2 = "solar"
url_theme1 = dbc.themes.MINTY
url_theme2 = dbc.themes.SOLAR

tab_card = {'height':'100%'} # all cards in the dash's layout will be at 100% of a given line

main_config = {  # basic config for all plotly's graphs
'hovermode':'x unified',
'legend':{'yanchor':'top',
            'y':0.9,
            'xanchor':'left',
            'x':0.1,
            'title':{'text': None},
            'font':{'color':'white'},
            'bgcolor':'rgba(0,0,0,0.5)'},
'margin':{'l':0,'r':0,'t':10,'b':0}
}

# ========== Dataset ETL ============ #
df_gas = pd.read_csv("data//combustiveis-estados.csv")
df_gas =df_gas.apply(lambda x: x.replace({
    'NORTE':'NORTH','SUL':'SOUTH','NORDESTE':'NORTHEAST',
    'SUDESTE':'SOUTHEAST','CENTRO OESTE':'MIDWEST' 
},regex = True))

df_min_wage = pd.read_excel("data//Min_Wage_BR.xlsx")

df_gas = df_gas[['referencia','ano','mes','regiao','estado',
                'gasolina_comum_preco_revenda_avg',
                'etanol_hidratado_preco_revenda_avg',
                'oleo_diesel_preco_revenda_avg']]

df_gas['referencia'] = pd.to_datetime(df_gas['referencia'])

df_gas = df_gas.sort_values(by='referencia',ascending=True)

df_gas.rename(columns= {'referencia':'date','ano':'year','mes':'month','regiao':'region','estado':'state',
                    'gasolina_comum_preco_revenda_avg':'gasoline',
                        #'avg gas price (R$/L)',
                    'etanol_hidratado_preco_revenda_avg': 'ethanol',
                        #'avg ethanol price (R$/L)',
                    'oleo_diesel_preco_revenda_avg':'diesel'
                        #'avg diesel price (R$/L)'
                            },
                    inplace= True
                    )

df_gas = df_gas.dropna().reset_index(drop=True)

not_melt_cols = ['date','year','month','region','state']

melt_cols = ['gasoline','ethanol','diesel']

df_gas_melt = pd.melt(df_gas, id_vars=not_melt_cols,value_vars=melt_cols)

df_gas_melt.rename(columns = {'variable':'fuel_type','value':'price (R$/L)'},inplace = True)

df_gas_melt = pd.merge(df_gas_melt, df_min_wage, left_on='date',right_on='date')


df_store = df_gas_melt.to_dict() # transforming the df so we can use the dcc.Store methot on for the dash layout 

#========== test #============


# ========== Building the dash ============ #

app.layout = dbc.Container(children=[
                               dcc.Store(id='dataset',data=df_store),
                               dcc.Store(id='dataset_fixed',data=df_store), # just a backup df that will not be affected by outside filters
                               
                            # === Layout ===
                               
                               # --- 1st Row ---
                               dbc.Row([
                                   dbc.Col([
                                       dbc.Card([
                                           dbc.CardBody([
                                               dbc.Row([
                                                   dbc.Col([
                                                       html.Legend("Brazil's Fuel Prices")
                                                   ],sm=8),
                                                   dbc.Col([
                                                       html.I(className='fa fa-gas-pump', style={'font-size':'300%'})
                                                   ],sm=4,align = 'center')
                                               ]),
                                               dbc.Row([
                                                   dbc.Col([
                                                       ThemeSwitchAIO(aio_id='theme',themes = [url_theme1,url_theme2]),
                                                                      html.Legend("Created by Lucas Zarpellon")
                                                   ])
                                               ],style={'margin-top':'10px'}),
                                               dbc.Row([
                                                   dbc.Col([
                                                       dbc.Button("LinkedIn",
                                                                   #html.I(className='fa fa-github', style={'font-size':'300%'}),
                                                                   id='linkedin_link',href='https://www.linkedin.com/in/lucasingleszarpellon/', target='_blank' 
                                                                  )
                                                   ])
                                               ],style={'margin-top':'10px'}),
                                            dbc.Row([
                                                   dbc.Col([
                                                       dbc.Button("GitHub", href='https://www.linkedin.com/in/lucasingleszarpellon/', target='_blank')                                                       
                                                   ])
                                               ],style={'margin-top':'10px'}),
                                            dbc.Row([
                                                   dbc.Col([
                                                       dbc.Button("Portfolio", href='https://www.linkedin.com/in/lucasingleszarpellon/', target='_blank')                                                       
                                                   ])
                                               ],style={'margin-top':'10px'})
                                           ])
                                       ],style= tab_card)
                                   ], sm=4,lg=2),
                                   
                                   ## Graph 1 - r1-maxmin - shows the max and min prices ##
                                   
                                   dbc.Col([
                                       dbc.Card([
                                           dbc.CardBody([
                                               dbc.Row([
                                                   dbc.Col([
                                                       html.H3('Max and Min Prices'),
                                                       html.H6('Fuel Type:'),
                                                       dcc.Dropdown(
                                                           id='filetype_filter1',
                                                           value=df_gas_melt.at[df_gas_melt.index[1],'fuel_type'],
                                                           clearable= False,
                                                           className= 'dbc',
                                                           options=[
                                                               {'label':x,'value':x} for x in df_gas_melt.fuel_type.unique()
                                                           ])                              
                                                    ],sm=6)
                                                   ]),
                                               dbc.Row([
                                                   dbc.Col([
                                                       
                                                       dcc.Graph(id='r1-maxmin',config={'displayModeBar':False,'showTips':True})
                                                   ])
                                               ])
                                           ])
                                       ],style=tab_card)
                                   ],sm=8,lg=3),
                                   # row 1 Col 2 filters
                                   dbc.Col([
                                       dbc.Card([
                                           dbc.CardBody([
                                               dbc.Row([
                                                   dbc.Col([
                                                       html.H6('Year:'),
                                                       dcc.Dropdown(
                                                           id='year_filter',
                                                           value=df_gas_melt.at[df_gas_melt.index[1],'year'],
                                                           clearable= False,
                                                           className= 'dbc',
                                                           options=[
                                                               {'label':x,'value':x} for x in df_gas_melt.year.unique()
                                                           ])                              
                                                    ],sm=6),
                                                   dbc.Col([
                                                       html.H6('Region:'),
                                                       dcc.Dropdown(
                                                           id='region_filter',
                                                           value=df_gas_melt.at[df_gas_melt.index[1],'region'],
                                                           clearable= False,
                                                           className= 'dbc',
                                                           options=[
                                                               {'label':x,'value':x} for x in df_gas_melt.region.unique()
                                                           ])                              
                                                    ],sm=6)
                                               ]),
                                               
                                               
                                               
                                               dbc.Row([
                                                # Row 1 regional Graph
                                                   dbc.Col([
                                                       dcc.Graph(id='r1-regionalgraph',config={'displayModeBar':False,'showTips':True})
                                                   ],sm=12,md=6),
                                                # Row 1 state Graph
                                                   dbc.Col([
                                                       dcc.Graph(id='r1-stategraph',config={'displayModeBar':False,'showTips':True})
                                                   ],sm=12,md=6)
                                               ], style={'column-gap': '0px'})
                                           ])
                                       ],style=tab_card)
                                   ],sm=12,lg=7)                                   
                               ],className='g-2 my-auto'), # makes the cards equaly spaced 
                               
                               # --- 2nd Row ---
                               dbc.Row([
                                    dbc.Col([
                                       dbc.Card([
                                           dbc.CardBody([
                                               html.H3('Fuel Type Price Comparison'),
                                               dbc.Row([
                                                   dbc.Col([
                                                       html.H6('State:'),
                                                       dcc.Dropdown(
                                                           id='state_filter1',
                                                           value=df_gas_melt.at[df_gas_melt.index[1],'state'],                                                          
                                                           clearable= False,
                                                           className= 'dbc',
                                                           options=[
                                                               {'label':x,'value':x} for x in df_gas_melt.state.unique()
                                                           ]),                                                                   
                                                   ], sm=10, md=5),
                                               ],style={'magin-top': '20px'}),
                                               dcc.Graph(id='direct_comparison_graph',config={'displayModeBar':False,'showTips':False})#,
                                               #html.P(id='desc_comparison',style={'color':'gray','font-size':'80%'}),
                                           ])
                                       ],style=tab_card)
                                   ],sm=12,md=6,lg=4),
                                                                       
                                   dbc.Col([
                                       dbc.Card([
                                           dbc.CardBody([
                                               html.H3('Price per State'),
                                               html.H6('State:'),
                                               dbc.Row([
                                                   dbc.Col([
                                                       dcc.Dropdown(
                                                           id='state_filter',
                                                           value=[
                                                                df_gas_melt.at[df_gas_melt.index[3],'state'],
                                                                df_gas_melt.at[df_gas_melt.index[13],'state']
                                                           ],
                                                           clearable= False,
                                                           className= 'dbc',
                                                           multi=True,
                                                           options=[
                                                               {'label':x,'value':x} for x in df_gas_melt.state.unique()
                                                           ]),      
                                                                                                                    
                                                   ], sm=10),
                                                   dbc.Col([
                                                       html.H6('Fuel Type:'),
                                                        dcc.Dropdown(
                                                            id='fuel_filter',
                                                            value=df_gas_melt.at[df_gas_melt.index[1],'fuel_type'],                                                          
                                                            clearable= False,
                                                            className= 'dbc',
                                                            options=[
                                                                {'label':x,'value':x} for x in df_gas_melt.fuel_type.unique()
                                                            ]),                                                                   
                                                   ],sm=10, md=5),
                                               ]),
                                               dbc.Row([
                                                   dbc.Col([
                                                       dcc.Graph(id='r2-trend',config={'displayModeBar':False,'showTips':True})
                                                   ])
                                               ])
                                               
                                           ])
                                       ],style=tab_card)
                                   ],sm=12,md=6,lg=5),

                                dbc.Col([
                                    
                                    dbc.Row([
                                        dbc.Col([
                                            html.H3('Year comparison'),
                                            dcc.RangeSlider(
                                                id='year_sclicer',
                                                marks = None,
                                                step=3,
                                                min=2001,
                                                max=2024,
                                                value=[2023,2022],
                                                dots=True,
                                                pushable=1,
                                                tooltip = {
                                                        'placement':'bottom',
                                                        'always_visible':True,
                                                        'style':{'color':'LightSteelBlue','fontSize':'15px'}
                                                },
                                            )
                                        ]),                                        
                                    ],justify='center',style={'padding-bottom':'15px','height':'50%'}),
                                    
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardBody([
                                                    dcc.Graph(id='card_kpi1',config={'displayModeBar':False,'showTips':False},style={'margin-top':'30px'})
                                                ])
                                            ],style=tab_card)
                                        ])
                                    ],justify='center',style={'padding-bottom':'7px','height':'50%'}),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardBody([
                                                    html.H3('Relative minimal wage for comparison in R$'),
                                                    dcc.Graph(id='card_kpi2',config={'displayModeBar':False,'showTips':False},style={'margin-top':'30px'})
                                                ])
                                            ],style=tab_card)
                                        ])
                                    ],justify='center',style={'height':'50%'})
                                ],sm=12,lg=3,style={'height':'100%'})                                   
                               ],className='g-2 my-auto')
                               
                               
                               
                           ],fluid=True, style={'height': '100%'})
                           
                           
# ========== Dash Callbacks ========== #

# --- 1st Row Callbacks --- #


## Max Min Graph ##

@app.callback(
    Output('r1-maxmin','figure'),
    [Input('dataset_fixed','data'),
    Input('filetype_filter1','value'),
    Input(ThemeSwitchAIO.ids.switch('theme'),'value')]
)

def func(data,fueltype,toggle):
    template = template_theme1 if toggle else template_theme2
    
    df = pd.DataFrame(data)

    df_max = df.groupby(['year','fuel_type'])['price (R$/L)'].max()
    df_min = df.groupby(['year','fuel_type'])['price (R$/L)'].min()

    df_maxmin = pd.concat([df_max,df_min], axis = 1)
    df_maxmin.columns = ['max','min']
    df_maxmin = df_maxmin.reset_index()

    df_maxmin_melt = pd.melt(df_maxmin, id_vars=['year','fuel_type'], value_vars=['max','min'])
    df_maxmin_melt = df_maxmin_melt[df_maxmin_melt.fuel_type.isin([fueltype])]

    fig = px.line(df_maxmin_melt,x='year', y='value',color = 'variable',template=template)
    fig.update_layout(main_config,height=150,xaxis_title=None,yaxis_title=None)

    return fig

## Region&State Graphs ##
@app.callback(
    [Output('r1-regionalgraph', 'figure'),
    Output('r1-stategraph', 'figure')],
    [Input('dataset_fixed', 'data'),
    Input('year_filter', 'value'),
    Input('region_filter', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)
def graph1(data, year, region,toggle):
    template = template_theme1 if toggle else template_theme2

    df = pd.DataFrame(data)
    df_yearly = df[df.year.isin([year])]

    df_region = df_yearly.groupby(['year','fuel_type','region'])['price (R$/L)'].mean().reset_index()
    df_state = df_yearly.groupby(['year', 'fuel_type','state', 'region'])['price (R$/L)'].mean().reset_index()
    df_state = df_state[df_state.region.isin([region])]

    df_region['price (R$/L)'] = df_region['price (R$/L)'].round(decimals = 2)
    df_state['price (R$/L)'] = df_state['price (R$/L)'].round(decimals = 2)

    fig1_text = {region: {fuel_type: f'{region} - {fuel_type} - R${price:.2f}/L' for fuel_type, price in zip(group['fuel_type'], group['price (R$/L)'])} for region, group in df_region.groupby('region')}
    fig2_text = {state: {fuel_type: f'{state} - {fuel_type} - R${price:.2f}/L' for fuel_type, price in zip(group['fuel_type'], group['price (R$/L)'])} for state, group in df_state.groupby('state')}

    fig1 = go.Figure()
    for fuel_type in df_region['fuel_type'].unique():
        df_filtered  = df_region[df_region['fuel_type'] == fuel_type]
        fig1.add_trace(go.Bar(
            x=df_filtered ['price (R$/L)'],
            y=df_filtered ['region'],
            name=fuel_type,
            orientation='h',
            text=[fig1_text[region][fuel_type] for region in df_filtered['region']],
            textposition='auto',
            insidetextanchor='end',
            insidetextfont=dict(family='Times', size=15),
            showlegend=False
    ))
        
    fig2 = go.Figure()
    for fuel_type in df_state['fuel_type'].unique():
        df_filtered  = df_state[df_state['fuel_type'] == fuel_type]
        fig2.add_trace(go.Bar(
            x=df_filtered ['price (R$/L)'],
            y=df_filtered ['state'],
            name=fuel_type,
            orientation='h',
            text=[fig2_text[state][fuel_type] for state in df_filtered['state']],
            insidetextanchor='end',
            insidetextfont=dict(family='Times', size=15),
            showlegend=False 
    ))

    fig1.update_layout(
        main_config,barmode='stack', 
        yaxis={'showticklabels':False}, 
        height=200, 
        template=template,
        xaxis=dict(
            autorange='reversed'
        )
    )
    
    fig2.update_layout(
        main_config,barmode='stack', 
        yaxis={'showticklabels':False}, 
        height=200, 
        template=template
    )

    return [fig1, fig2]


# --- 2nd Row Callbacks --- #

## Fuel comparison ##

@app.callback(
    [Output('direct_comparison_graph', 'figure')],
    [Input('dataset_fixed', 'data'),
    Input('state_filter1', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)
def func(data, state, toggle):
    
    template = template_theme1 if toggle else template_theme2
    dff = pd.DataFrame(data)
    mask = dff.state.isin([state])
    
    fig = px.line(dff[mask], x='date', y='price (R$/L)',color='fuel_type', template=template)
    
    fig.update_layout(main_config, height=425, xaxis_title=None)

    return [fig]

## state comparison ##

@app.callback(
    Output('r2-trend', 'figure'),
    [Input('dataset', 'data'), 
    Input('state_filter', 'value'),
    Input('fuel_filter', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)
def animation(data, state, fueltype,toggle):
    template = template_theme1 if toggle else template_theme2

    dff = pd.DataFrame(data)
    dff = dff[dff.fuel_type.isin([fueltype])]
    mask = dff.state.isin(state)
    
    
    fig = px.line(dff[mask], x='date', y='price (R$/L)',color='state', template=template)
    fig.update_layout(main_config, height=425, xaxis_title=None)

    return fig

## KPI card 1 ##

@app.callback(
    Output("card_kpi1", "figure"),
    [Input('dataset', 'data'), 
    Input('fuel_filter', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)
def card1(data, fueltype, toggle):
    template = template_theme1 if toggle else template_theme2

    dff = pd.DataFrame(data)
    df_final = dff[dff.fuel_type.isin([fueltype])]

    data1 = str(int(dff.year.min()) - 1)
    data2 = dff.year.max()   
    
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode = "number+delta",
        title = {"text": f"<span style='size:60%'>{fueltype}</span><br><span style='font-size:0.7em'>{data1} - {data2}</span>"},
        value = df_final.at[df_final.index[-1],'price (R$/L)'],
        number = {'prefix': "R$", 'valueformat': '.2f'},
        delta = {'relative': True, 'valueformat': '.1%', 'reference': df_final.at[df_final.index[0],'price (R$/L)']}
    ))
    
    fig.update_layout(main_config, height=250, template=template)
    
    return fig

## KPI card 2 ##

@app.callback(
    Output("card_kpi2", "figure"),
    [Input('dataset', 'data'), 
    Input('fuel_filter', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)
def card1(data, fueltype, toggle):
    template = template_theme1 if toggle else template_theme2

    dff = pd.DataFrame(data)
    df_final = dff[dff.fuel_type.isin([fueltype])]

    data1 = str(int(dff.year.min()) - 1)
    data2 = dff.year.max()   
    
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode = "number+delta",
        title = {"text": f"<span style='size:60%'>{fueltype}</span><br><span style='font-size:0.7em'>{data1} - {data2}</span>"},
        value = df_final.at[df_final.index[-1],'Permonth'],
        number = {'prefix': "R$", 'valueformat': '.2f'},
        delta = {'relative': True, 'valueformat': '.1%', 'reference': df_final.at[df_final.index[0],'Permonth']}
    ))
    
    fig.update_layout(main_config, height=250, template=template)
    
    return fig

## date slider ##

@app.callback(
    Output('dataset', 'data'),
    [Input('year_sclicer', 'value'),
    Input('dataset_fixed', 'data')], prevent_initial_call=True
)
def range_slider(range, data):
    dff = pd.DataFrame(data)
    dff = dff[(dff['date'] >= f'{range[0]}-01-01') & (dff['date'] <= f'{range[1]}-31-12')]
    data = dff.to_dict()

    return data

# Server Run
if __name__ == '__main__':
    app.run_server(debug = False)






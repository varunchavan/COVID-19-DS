import pandas as pd
import numpy as np
import os,sys
import dash

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State

PATH_OF_DATA=os.path.join(os.path.dirname(__file__),r'..\data')
Directory_Path=os.path.join(os.path.dirname(__file__),r'..\features')
Model_Path=os.path.join(os.path.dirname(__file__),r'..\models')


sys.path.insert(0, PATH_OF_DATA)
sys.path.insert(0, Directory_Path)
sys.path.insert(0, Model_Path)
from SIR_methods import SIR_modelling

import plotly.graph_objects as go
from scipy import optimize
from scipy import integrate


#print(os.getcwd())

Directory_path=os.path.join(os.path.dirname(__file__),r'..\..\data\raw\COVID-19')
csv_directory_path=os.path.join(Directory_path,r'..\..\processed' )
csv_path_1=os.path.join(csv_directory_path,'COVID_final_set.csv')
df_input_large = pd.read_csv(csv_path_1, sep = ';')

fig = go.Figure()
app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''

    # Task 2: SIR Virus spread model implemented in Dynamic Dashboard 
    
    
    Select any country that needs to be visualize

    '''),


    dcc.Dropdown(
        id = 'country_drop_down',
        options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
        value= 'France', 
        multi=False),

    dcc.Graph(figure = fig, id = 'SIR_graph')
    ])

#def SIR(countries):
#
#    SIR_modelling()


@app.callback(
    Output('SIR_graph', 'figure'),
    [Input('country_drop_down', 'value')])

def update_SIR_figure(country_drop_down):

    traces = []

    df_plot = df_input_large[df_input_large['country'] == country_drop_down]
    df_plot = df_plot[['state', 'country', 'confirmed','confirmed_filtered', 'date']].groupby(['country', 'date']).agg(np.sum).reset_index()
    df_plot.sort_values('date', ascending = True).head()
    df_plot = df_plot.confirmed[40:]

    t, fitted = SIR_modelling(df_plot)

    traces.append(dict (x = t,
                        y = fitted,
                        mode = 'markers',
                        opacity = 1.1,
                        name = 'SIR-model-curve')
                  )

    traces.append(dict (x = t,
                        y = df_plot,
                        mode = 'lines',
                        opacity = 0.7,
                        name = 'Original Data curve')
                  )

    return {
            'data': traces,
            'layout': dict (
                width=1200,
                height=700,
                title = 'SIR model fitting',

                xaxis= {'title':'Days',
                       'tickangle':-48,
                        'nticks':21,
                        'tickfont':dict(size=16,color="#7f7f7f"),
                      },

                yaxis={'title': "Infected population"}
        )
    }


if __name__ == '__main__':
    app.run_server(debug = True, use_reloader = False)

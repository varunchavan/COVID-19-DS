import pandas as pd
import numpy as np
import subprocess
import dash,sys
import os
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State

import plotly.graph_objects as go

from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)


from scipy import signal


PATH_OF_DATA=os.path.join(os.path.dirname(__file__),r'..\data')
Directory_Path=os.path.join(os.path.dirname(__file__),r'..\features')
Model_Path=os.path.join(os.path.dirname(__file__),r'..\models')


sys.path.insert(0, PATH_OF_DATA)
sys.path.insert(0, Directory_Path)
sys.path.insert(0, Model_Path)


from GET_DATA import pull_JH_data
from PROCESSED_REL_JH_DATA import Relational_JH_data_store
from REL_PD_LARGE import Enlarged_pd_result
from build_features import *


pull_JH_data()
Relational_JH_data_store()
Enlarged_pd_result()

print(os.getcwd())
Directory_path=os.path.join(os.path.dirname(__file__),r'..\..\data\raw\COVID-19')
csv_directory_path=os.path.join(Directory_path,r'..\..\processed' )
csv_path_1=os.path.join(csv_directory_path,'COVID_final_set.csv')
df_input_large=pd.read_csv(csv_path_1,sep=';')


fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    #  Task 1: Covid-19 Prototyping (Dashboard) [Enterprise Data Science]

    '''),

    dcc.Markdown('''
    ## Choose country that you wish to visualize
    '''),


    dcc.Dropdown(
        id='country_drop_down',
        options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
        value=['Norway', 'Germany','US'], 
        multi=True
    ),

    dcc.Markdown('''
        ## Choose timeline of confirmed COVID-19 cases or approximated doubling time
        '''),


    dcc.Dropdown(
    id='doubling_time',
    options=[
        {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
        {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
        {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
        {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
    ],
    value='confirmed',
    multi=False
    ),

    dcc.Graph(figure=fig, id='main_window_slope')
])



@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('doubling_time', 'value')])


def update_plot(list_of_countries,doubling):


    if 'confirmed' == doubling:
        my_yaxis={'type':"log",
               'title':'Confirmed infected people (JH csse, log scale)'
              }
    elif 'confirmed_filtered' == doubling:
        my_yaxis={'type':"log",
               'title':'Timeline confirmed filtered'
              }
    elif 'confirmed_DR' == doubling:
        my_yaxis={'type':"log",
               'title':'doubling rate'
              }
    elif 'confirmed_filtered_DR' == doubling:
        my_yaxis={'type':"log",
               'title':'filtered doubling'
              }


    traces = []
    for each in list_of_countries:

        df_plot=df_input_large[df_input_large['country']==each]

        if doubling=='doubling_rate_filtered':
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.mean).reset_index()
        else:
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()
       #print(doubling)


        traces.append(dict(x=df_plot.date,
                                y=df_plot[doubling],
                                mode='markers+lines',
                                opacity=0.8,
                                name=each
                        )
                )

    return {
            'data': traces,
            'layout': dict (
                width=1150,
                height=680,

                xaxis={'title':'Timeline',
                        'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },

                yaxis=my_yaxis
        )
    }



def doubling_time_using_regression(in_array):
    ''' Linear Regression is used to make approximation of doubling rate
   

        Parameters is followed;
        
        in_array : pandas.series

        get following as Returns;
        
        Doubling rate: double
    '''

    y = np.array(in_array)
    X = np.arange(-1,2).reshape(-1, 1)

    assert len(in_array)==3
    reg.fit(X,y)
    intercept=reg.intercept_
    slope=reg.coef_

    return intercept/slope


def savgol_filter_groupby_fun(df_input,column='confirmed',window=5):
    ''' Savgol Filter which can be used in groupby apply function (data structure kept)

        parameters:
       
        df_input : pandas.series
        column : str
        window : int
            used data points to calculate the filter result

        Returns:
      
        df_result: pd.DataFrame
            the index of the df_input has to be preserved in result
    '''

    degree=1
    df_result=df_input

    filter_in=df_input[column].fillna(0) 

    result=signal.savgol_filter_groupby_fun(np.array(filter_in),
                           window, # window size used for filtering
                           1)
    df_result[str(column+'_filtered')]=result
    return df_result

def rolling_reg_doub_approx(df_input,col='confirmed'):
    ''' Here Rolling Regression used to make approximation of the doubling time'

        Parameters as followed;
        
        df_input: pd.DataFrame
        col: str
            defines the used column
        gives as following Returns;
        
        result: pd.DataFrame
    '''
    days_back=3
    result=df_input[col].rolling(
                window=days_back,
                min_periods=days_back).apply(doubling_time_using_regression,raw=False)



    return result




def filtered_data_calc_savgol(df_input,filter_on='confirmed'):
    ''' Calculate savgol filter and return merged data frame

        Parameters as follows;
        
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        gives as Returns;
        
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in filtered_data_calc_savgol not all columns in data frame'

    df_output=df_input.copy() # It is copied here in order to prevent the filter_on column to be overwritten

    pd_filtered_result=df_output[['state','country',filter_on]].groupby(['state','country']).apply(savgol_filter_groupby_fun)#.reset_index()

   
    df_output=pd.merge(df_output,pd_filtered_result[[str(filter_on+'_filtered')]],left_index=True,right_index=True,how='left')
   
    return df_output.copy()





def doubling_rate_calculation(df_input,filter_on='confirmed'):
    ''' Calculate approximated doubling rate and return merged data frame

        Parameters as follows;
        
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        gives as follwing Returns;
        
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in filtered_data_calc_savgol not all columns in data frame'


    pd_DR_result= df_input.groupby(['state','country']).apply(rolling_reg_doub_approx,filter_on).reset_index()

    pd_DR_result=pd_DR_result.rename(columns={filter_on:filter_on+'_DR',
                             'level_2':'index'})

    #Here merge on the index of our big table and on the index column after groupby is done
    df_output=pd.merge(df_input,pd_DR_result[['index',str(filter_on+'_DR')]],left_index=True,right_on=['index'],how='left')
    df_output=df_output.drop(columns=['index'])


    return df_output



if __name__ == '__main__':

    app.run_server(port=8051,debug=True, use_reloader=False)

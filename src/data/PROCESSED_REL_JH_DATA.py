import pandas as pd
import numpy as np
import os
from datetime import datetime


def Relational_JH_data_store():
    ''' Transformes the COVID data in a relational data set

    '''
    dir_path=os.path.join(os.path.dirname(__file__),r'..\..\data\raw\COVID-19')
    data_path=os.path.join(dir_path,'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    pd_raw=pd.read_csv(data_path)

    pd_data_base=pd_raw.rename(columns={'Country/Region':'country',
                      'Province/State':'state'})

    pd_data_base['state']=pd_data_base['state'].fillna('no')

    pd_data_base=pd_data_base.drop(['Lat','Long'],axis=1)


    pd_relational_model=pd_data_base.set_index(['state','country']) \
                                .T                              \
                                .stack(level=[0,1])             \
                                .reset_index()                  \
                                .rename(columns={'level_0':'date',
                                                   0:'confirmed'},
                                                  )

    pd_relational_model['date']=pd_relational_model.date.astype('datetime64[ns]')
    csv_path=os.path.join(dir_path,r'..\..\processed\COVID_relational_confirmed.csv' )
    pd_relational_model.to_csv(csv_path,sep=';',index=False)
    print(' Number of rows stored: '+str(pd_relational_model.shape[0]))

if __name__ == '__main__':

    Relational_JH_data_store()

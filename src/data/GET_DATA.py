
import subprocess
import os

import pandas as pd
import numpy as np

from datetime import datetime

import requests
import json

def pull_JH_data():
    ''' Get data by a git pull request, the source code has to be pulled first
        Result is stored in the predifined csv structure
    '''
    dir_path=os.path.join(os.path.dirname(__file__),r'..\..\data\raw\COVID-19')
    print(dir_path)
    git_pull=subprocess.Popen('git init & git reset --hard & git pull https://github.com/CSSEGISandData/COVID-19.git' ,cwd=dir_path,shell=True,
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out,error)=git_pull.communicate()
    print('ERROR:', str(error))
    print('out:', str(out))


    #print("Error : " + str(error))
    #print("out : " + str(out))


def get_current_data_germany():
    ''' Get current data from germany, attention API endpoint not too stable
        Result data frame is stored as pd.DataFrame

    '''
    # 16 states
    #data=requests.get('https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/Coronaf%C3%A4lle_in_den_Bundesl%C3%A4ndern/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json')

    # 400 regions / Landkreise
    data=requests.get('https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_Landkreisdaten/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json')

    json_object=json.loads(data.content)
    full_list=[]
    for pos,each_dict in enumerate (json_object['features'][:]):
        full_list.append(each_dict['attributes'])

    pd_full_list=pd.DataFrame(full_list)
    pd_full_list.to_csv('data/raw/NPGEO/GER_state_data.csv',sep=';')
    print(' Number of regions rows: '+str(pd_full_list.shape[0]))

if __name__ == '__main__':
    pull_JH_data()
    get_current_data_germany()

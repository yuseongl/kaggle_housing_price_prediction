import pandas as pd
import numpy as np
from imblearn.under_sampling import TomekLinks, OneSidedSelection, RandomUnderSampler
from imblearn.combine import SMOTETomek

class Imbalance_Module:
    '''
    function for resampli
    '''
    def __init__(self):
        #self.ada = TomekLinks(sampling_strategy='majority')
        #self.ada = RandomUnderSampler(sampling_strategy='majority', random_state=2023)
        self.ada = OneSidedSelection(sampling_strategy='majority', random_state=2023)
        #self.ada = SMOTETomek(sampling_strategy='auto')
        
    def resample(self,df):
        print('resampling start!')
        y = df['ECLO']
        for i in range(3):
            df, y = self.ada.fit_resample(df, y)
        #X_res, y_res = self.ada.fit_resample(X_res, y_res.squeeze())
        #X_res, y_res = self.ada.fit_resample(X_res, y_res.squeeze())
        #X_res, y_res = self.ada.fit_resample(X_res, y_res.squeeze())
        #X_res, y_res = self.ada.fit_resample(X_res, y_res.squeeze())
        
        return df
       
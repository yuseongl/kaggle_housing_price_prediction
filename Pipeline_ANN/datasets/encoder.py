import pandas as pd
import numpy as np
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

class Encoder_Module:
    def __init__(self,y:pd.DataFrame=None):
        self.label = LabelEncoder()
        self.onehot = OneHotEncoder(dtype=np.float32, sparse_output=False, drop='if_binary', handle_unknown='ignore')
        self.target = TargetEncoder()
        self.y = y
        
    def encoder(self, df:pd.DataFrame, enc:str='onehot'):
        if enc == 'label':
            df.loc[:,:] = df.loc[:,:].apply(self.label.fit_transform)
        elif enc == 'onehot':
            df = pd.DataFrame(self.onehot.fit_transform(df))
        elif enc == 'target':
            df.loc[:,:] = df.loc[:,:].apply(self.target.fit_transform, self.y)
        
        return df
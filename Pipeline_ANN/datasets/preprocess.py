import pandas as pd
from .encoder import Encoder_Module
from .external import external_data

class preprosess_Module:
    def __init__(self,df:pd.DataFrame):
        self.df = df
    
    def preprocess(self, df:pd.DataFrame, df_tst:pd.DataFrame, features:iter=None):
        df.dropna(axis=0, subset=['SalePrice'], inplace=True)
        #df.drop(['SalePrice'], axis=1, inplace=True)
        
        df_num = df.select_dtypes(exclude=['object'])
        df_tst_num = df_tst.select_dtypes(exclude=['object'])
        
        missing_df = (df_num.isnull().sum())
        
        all_columns = df_num.columns
        df_num = df_num.drop(all_columns[missing_df > 0], axis=1)
        df_tst_num = df_tst_num.drop(all_columns[missing_df > 0], axis=1)
        
        # 결측치 저리
        df_num = df_num.fillna(df_num.min())
        df_tst_num = df_tst_num.fillna(df_tst_num.min())
        
        df_cat = df.select_dtypes(include=['object'])
        df_cat_tst = df_tst.select_dtypes(include=['object'])
        
        #df_cat.fillna(0, inplace=True)
        #df_cat_tst.fillna(0, inplace=True)
        encoder = Encoder_Module(df['SalePrice'])
        df_cat = encoder.encoder(df_cat)
        df_cat_tst = encoder.encoder(df_cat_tst)
        df = pd.concat([df_num, df_cat], axis=1)
        df_tst = pd.concat([df_tst_num, df_cat_tst], axis=1)
        
        df_cat.to_csv('data.csv')
        
        df = pd.concat([df_cat, df_num], axis=1)
        df_tst = pd.concat([df_cat_tst, df_tst_num], axis=1)
        return df, df_tst
    
    def __call__(self, df:pd.DataFrame):
        return
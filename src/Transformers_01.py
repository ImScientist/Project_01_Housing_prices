import matplotlib.pyplot as plt

from sklearn import base, utils
import pandas as pd
import numpy as np
import copy

from sklearn.metrics import confusion_matrix




class my_AppCopy(base.BaseEstimator, base.TransformerMixin):
    '''
    copy the relevant columns
    turn the columns with type=int to columns with type=float
    '''
    def __init__(self, relevant_cols):
        self.relevant_cols = relevant_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #X = copy.deepcopy(X[self.relevant_cols])
        X = X[self.relevant_cols].copy(deep=True)
        
        #cols_int = [ col for col in X.columns.values if (np.result_type(X[col])=='int32' or np.result_type(X[col])=='int64')]
        #for col in cols_int:
        #    X[col] = X[col].astype(float)
            
        return X

    

class my_Dummyrizer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, relevant_cols):
        self.relevant_cols = relevant_cols
        self.unique_values_for_every_col = {}
        self.all_cols = None

    def fit(self, X, y=None):
        for col in self.relevant_cols:
            self.unique_values_for_every_col[col] = X[col].fillna(value='nan').unique()
        return self

    def transform(self, X):
        for col in self.relevant_cols:
            
			# TODO add astype(int)
            for value in self.unique_values_for_every_col[col]:
                X.loc[:,col + '_' + str(value)] = X[col].fillna(value='nan').apply( lambda x: 1 if x==value else 0)
                
            X = X.drop([col], axis=1)

        self.all_cols = X.columns.values
        return X



class my_StdScaler(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, relevant_cols):
        self.relevant_cols = relevant_cols
        self.maps = {}
        self.all_cols = None

    def fit(self, X, y=None):
        
        for col in self.relevant_cols:
            if col in X.columns:
                self.maps[col]={}
                self.maps[col]['mean'] = X[col].mean()
                self.maps[col]['std']  = X[col].std()
        
        return self

    def transform(self, X):
        
        for col in self.relevant_cols:
            if col in X.columns:
                X[col] =  (X[col]-self.maps[col]['mean'])/self.maps[col]['std']
        
        self.all_cols = X.columns.values
        return X



class my_CreateNewFeatures(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['LotArea_60000']         = 1.0*(X['LotArea'] > 60000)
        X['YearBuilt_2000']        = 1.0*(X['YearBuilt'] > 1999)
        X['2ndFlrSF_missing']      = 1.0*(X['2ndFlrSF'] == 0)
        X['LowQualFinSF_missing']  = 1.0*(X['LowQualFinSF']==0)
        X['EnclosedPorch_missing'] = 1.0*(X['EnclosedPorch']==0)
        X['OpenPorchSF_missing']   = 1.0*(X['OpenPorchSF']==0)
        X['WoodDeckSF_missing']    = 1.0*(X['WoodDeckSF']==0)
        X['3SsnPorch_missing']     = 1.0*(X['3SsnPorch']==0)
        X['ScreenPorch_missing']   = 1.0*(X['ScreenPorch']==0)
        
        #X               = X.drop(['LotFrontage','MasVnrArea','GarageYrBlt', 'GarageCars', 'TotRmsAbvGrd'], axis=1) 
        # the first 3 have nan-values the last 2 are highly correlated with other features
        
        return X



		
class my_DropFeatures(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, relevant_cols):
        self.relevant_cols = relevant_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.drop(self.relevant_cols, axis=1) 
        return X



class my_log1p_trf(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, relevant_cols):
        self.relevant_cols = relevant_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.relevant_cols:
            if col in X.columns:
                X[col] = np.log1p(X[col])
            
        return X
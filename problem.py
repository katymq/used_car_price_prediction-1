import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit, KFold

problem_title = 'Used Cars Price Prediction'
_target_column_name = 'price'

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorRegressor()

class PRICE_error(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf
    
    def __init__(self, name='price error',precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
            
        true = np.log10(np.maximum(1., y_true))
        pred = np.log10(np.maximum(1., y_pred))
        loss = np.mean(np.abs(true - pred))
    
        return loss

score_types = [
    PRICE_error(name='price error', precision=2),
]

def get_cv(X, y):
    cv = KFold(n_splits=8, random_state=45)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False, compression='zip')
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv.zip'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv.zip'
    return _read_data(path, f_name)
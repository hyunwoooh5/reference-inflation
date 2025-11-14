import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateToNumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, base_date='1995-01-01', column='preprint_date'):
        self.base_date = pd.to_datetime(base_date)
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        date_col = X[self.column]

        dates = pd.to_datetime(date_col, format='mixed', errors='coerce')
        numeric_dates = (dates - self.base_date) / np.timedelta64(1, 'D')

        return numeric_dates.values.reshape(-1, 1)

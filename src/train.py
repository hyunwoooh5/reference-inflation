import pandas as pd
import xgboost as xgb
import pickle
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from util import DateToNumericTransformer


def train(df, seed=42):
    y = df['number_of_references'].to_numpy()
    X = df.drop('number_of_references', axis=1)

    numeric_features = ['number_of_pages', 'author_count']
    categorical_features = ['document_type', 'publication_type']
    date_features = ['preprint_date']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))  # avoid NaN
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # avoid NaN
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    date_transformer = Pipeline(steps=[
        ('date_to_num', DateToNumericTransformer(
            base_date='1995-01-01', column='preprint_date')),
        ('imputer', SimpleImputer(strategy='median'))  # avoid NaN
    ])

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('date', date_transformer, date_features)
        ],
        remainder='drop'  # Unnecessary features are dropped
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('xgboost',
         xgb.XGBRegressor(
             n_estimators=28,
             max_depth=4,
             learning_rate=0.32644423647442644,
             random_state=seed,
             n_jobs=-1
         )

         )


    ])
    pipeline.fit(X, y)

    return pipeline


if __name__ == "__main__":
    df = pd.read_csv("data/data_nucl-th_100_cleaned.csv")
    df = df.drop(columns=[
                 'id', 'citation_count_without_self_citations', 'citation_count', 'refereed'])

    model = train(df)

    with open("bin/model.bin", "wb") as f:
        pickle.dump(model, f)

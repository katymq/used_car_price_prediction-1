from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        numeric_transformer = Pipeline(steps=[('impute', SimpleImputer(strategy='median'))])
        
        model_te = X_df[["model"]].dropna()
        model_te = pd.concat((model_te, pd.Series(y_array)), axis=1)
        model_te = model_te.rename(columns={0:"price"})
        model_te = model_te.groupby("model").agg(["median"])
        model_te["model_rank"] = model_te["price"]["median"].rank(ascending=False)
        model_te["model_te"] = model_te.index
        model_rank = pd.DataFrame({"model_te":model_te["model_te"], "model_rank":model_te["model_rank"]})


        brand_te = X_df[["brand"]].dropna()
        brand_te = pd.concat((brand_te, pd.Series(y_array)), axis=1)
        brand_te = brand_te.rename(columns={0:"price"})
        brand_te = brand_te.groupby("brand").agg(["median"])
        brand_te["brand_rank"] = brand_te["price"]["median"].rank(ascending=False)
        brand_te["brand_te"] = brand_te.index
        brand_rank = pd.DataFrame({"brand_te":brand_te["brand_te"], "brand_rank":brand_te["brand_rank"]})
        
        def process_date(X):
            dateCreated = pd.to_datetime(X['dateCreated'], format='%Y-%m-%d')
            return np.c_[dateCreated.dt.year, dateCreated.dt.month, dateCreated.dt.day]
        date_transformer = FunctionTransformer(process_date, validate=False)


        def process_zipcodes(X):
            zipcode_nums = pd.to_numeric(X['postalCode'].astype(str).str[:2], errors='coerce')
            return zipcode_nums.values[:, np.newaxis]
        zipcode_transformer = FunctionTransformer(process_zipcodes, validate=False)


        def process_gear(X):
            gear = X['gearbox']
            di_gear = {"manuell": 1, "automatik": 2}
            gear = gear.replace(di_gear)
            return pd.to_numeric(gear).values[:, np.newaxis]
        gear_transformer = FunctionTransformer(process_gear, validate=False)

        def process_seller(X):
            seller = X['seller']
            di_seller = {"privat": 1, "gewerblich": 2}
            seller = seller.replace(di_seller)
            return pd.to_numeric(seller).values[:, np.newaxis]
        seller_transformer = FunctionTransformer(process_seller, validate=False)

        def process_car_age(X):
            yearOfRegistration = X['yearOfRegistration']
            monthOfRegistration = X['monthOfRegistration']
            dateCreated = X['dateCreated']
            year_month_day_created = process_date(X)
            carAge = X.insert(1, 'carAge', yearOfRegistration - year_month_day_created[year], True)
            return pd.to_numeric(carAge).values[:, np.newaxis]
        car_age_transformer = FunctionTransformer(process_car_age, validate=False)
 
        def brand_model(X): 
            brand = pd.merge(X, brand_rank, left_on="brand", right_on="brand_te", how="left")
            return brand[["brand_rank"]]
        brand_transformer = FunctionTransformer(brand_model, validate=False)

        def process_model(X): 
            model = pd.merge(X, model_rank, left_on="model", right_on="model_te", how="left")
            return model[["model_rank"]]
        model_transformer = FunctionTransformer(process_model, validate=False)

        seller_col = ['seller']
        gear_col = ['gearbox']
        num_cols = ['powerPS', 'kilometer']
        zipcode_col = ['postalCode']
        model_col = ['model']
        brand_col = ['brand']
        drop_cols = ['dateCrawled', 'nrOfPictures', 'lastSeen']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('seller', make_pipeline(seller_transformer, SimpleImputer(strategy='median')), seller_col),
                ('gear', make_pipeline(gear_transformer, SimpleImputer(strategy='median')), gear_col),
                ('zipcode', make_pipeline(zipcode_transformer, SimpleImputer(strategy='median')), zipcode_col),
                ('num', numeric_transformer, num_cols),
                ('model_te', make_pipeline(model_transformer, SimpleImputer(strategy='median')), model_col),
                ('brand_te', make_pipeline(brand_transformer, SimpleImputer(strategy='median')), brand_col),
                ('drop cols', 'drop', drop_cols),
                ])
        
        self.preprocessor = preprocessor
        self.preprocessor.fit(X_df, y_array)
        return self
        
    def transform(self, X_df):
        return self.preprocessor.transform(X_df)
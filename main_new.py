from sklearn.model_selection import  StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import   StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import joblib
import os

MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"

def build_pipeline(num_attributes,cat_attributes):
  num_pipeline=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
  cat_pipeline=Pipeline([("encoding",OneHotEncoder(handle_unknown="ignore"))])
  Full_pipeline=ColumnTransformer([("num",num_pipeline,num_attributes),("cat",cat_pipeline,cat_attributes)])
  return Full_pipeline
 #fuction to return the pipeline
if not os.path.exists(MODEL_FILE):
  #lets train the model
  housing=pd.read_csv("housing.csv")
#create a stratified test set
  housing["income_cat"]=pd.cut(housing["median_income"],bins=[0.0,1.5,3.0,4.5,6,np.inf],labels=[1,2,3,4,5])
  split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
  for train_index,test_index in split.split(housing,housing["income_cat"]):
    housing.loc[test_index].drop("income_cat",axis=1).to_csv("input.csv",index=False)
    housing=housing.iloc[train_index].drop("income_cat",axis=1)
#Seperate target value and feature values
  housing_labels=housing["median_house_value"].copy()
  housing_features=housing.drop("median_house_value",axis=1)
  num_attributes=housing_features.drop("ocean_proximity",axis=1).columns.to_list()
  cat_attributes=["ocean_proximity"]
  pipeline=build_pipeline(num_attributes,cat_attributes)
  housing_prepared=pipeline.fit_transform(housing_features)
  print(housing_prepared)
  #train the model using random forest regressor
  model=RandomForestRegressor(random_state=42)
  model.fit(housing_prepared,housing_labels)
  joblib.dump(model,MODEL_FILE)
  joblib.dump(pipeline,PIPELINE_FILE)
  print("Congrats!Your model is trained")
else:
  model=joblib.load(MODEL_FILE)
  pipeline=joblib.load(PIPELINE_FILE)
  input_data=pd.read_csv("input.csv")
  transformed_input=pipeline.transform(input_data)
  predictions=model.predict(transformed_input)
  input_data["median_house_value"]=predictions
  input_data.to_csv("output.csv")
  print("Inference is Complete.Results saved to output.csv.Enjoy!")




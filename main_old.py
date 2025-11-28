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
 #load the dataset
housing=pd.read_csv("housing.csv")
#create a stratified test set
housing["income_cat"]=pd.cut(housing["median_income"],bins=[0.0,1.5,3.0,4.5,6,np.inf],labels=[1,2,3,4,5])
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    train_set=housing.iloc[train_index].drop("income_cat",axis=1)
    test_set=housing.iloc[test_index].drop("income_cat",axis=1)
#we will work on the copy of training set
housing=train_set.copy()
#Seperate target value and feature values
housing_labels=housing["median_house_value"].copy()
housing=housing.drop("median_house_value",axis=1)
#list the numerical and categorical data
num_attributes=housing.drop("ocean_proximity",axis=1).columns.to_list()
cat_attributes=["ocean_proximity"]
#making pipelines for numerical columns
num_pipeline=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
#categorical pipeline
cat_pipeline=Pipeline([("encoding",OneHotEncoder(handle_unknown="ignore"))])
#construct full pipeline
Full_pipeline=ColumnTransformer([("num",num_pipeline,num_attributes),("cat",cat_pipeline,cat_attributes)])
# Get the final Data
housing_prepared=Full_pipeline.fit_transform(housing)
# print(housing_prepared) you can print to check housing_prepared it will give numpy array
#Train the model 
#using linear regresion
linear_reg=LinearRegression()
linear_reg.fit(housing_prepared,housing_labels)
linear_preds=linear_reg.predict(housing_prepared)
linear_rsmes=-cross_val_score(linear_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
# print(f"The root mean squared error using linear regression model is {linear_rsme}")
print(pd.Series(linear_rsmes).describe())


#using decison tree  regresion
decision_reg=DecisionTreeRegressor()
decision_reg.fit(housing_prepared,housing_labels)
decision_preds=decision_reg.predict(housing_prepared)
# rand_rsme=root_mean_squared_error(housing_labels,rand_preds)
dec_rsmes= -cross_val_score(decision_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
# print(f"The root mean squared error using decision tree regression model is {dec_rsmes}")
print(pd.Series(dec_rsmes).describe())

#using random forest regession

rand_reg=RandomForestRegressor()
rand_reg.fit(housing_prepared,housing_labels)
rand_preds=rand_reg.predict(housing_prepared)
rand_rsmes=dec_rsmes= -cross_val_score(rand_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
# print(f"The root mean squared error using random forest regressor model is {rand_rsmes}")
print(pd.Series(rand_rsmes).describe())










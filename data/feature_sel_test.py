import sys
print("Python:", sys.version)
print("Testing imports...")
import pandas
print("pandas OK")
import numpy
print("numpy OK")
import sklearn
print("sklearn OK")
from sklearn.linear_model import LassoCV, ElasticNetCV, Lasso
print("LassoCV OK")
from sklearn.ensemble import RandomForestRegressor
print("RF OK")
from sklearn.inspection import permutation_importance
print("permutation_importance OK")
from sklearn.preprocessing import StandardScaler
print("StandardScaler OK")
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
print("TimeSeriesSplit OK")
print("All imports successful!")

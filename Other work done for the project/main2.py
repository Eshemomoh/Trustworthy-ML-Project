# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:22:07 2022
file for comparing SHAP algorithms
@author: lucky
"""
import xgboost
import shap

# train an XGBoost model
#X, y = shap.datasets.boston()
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X,y = housing.data, housing.target
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])
# visualize the first prediction's explanation with a force plot
shap.plots.force(shap_values[0])
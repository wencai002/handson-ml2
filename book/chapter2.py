import os
import tarfile
import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

Download_Root = "https://raw.githubusercontent.com/ageron/handson-ml2/master"
Folder_Path = os.path.join("datasets","housing")
Download_URL = Download_Root + "datasets/housing/housing.tgz"

# def fetch_housing_data(housing_url=Download_URL,housing_path = Folder_Path):
#     os.makedirs(housing_path, exist_ok=True)
#     tgz_path = os.path.join(housing_path, "housing.tgz")
#     urllib.request.urlretrieve(housing_url, tgz_path)
#     housing_tgz=tarfile.open(tgz_path)
#     housing_tgz.extractall(path=housing_path)
#     housing_tgz.close()

housing_path = "~/PycharmProjects/Handsonml2book/handson-ml2/datasets/housing/housing.csv"
housing = pd.read_csv(housing_path)
# print(housing.head())
# print(housing.info())
# print(housing.describe())
#
import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(16,10))
# plt.show()
###########################################
# split between train and test, dont touch test after
###########################################
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1, 2, 3, 4, 5])
# housing["income_cat"].hist()
# plt.show()

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#housing = strat_train_set.copy()
#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#              s=housing["population"]/100, label="population", figsize=(4,3),
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
#plt.legend()
#plt.show()

#corr_matrix = housing.corr()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], alpha=0.1, figsize=(4,3))
# plt.show()

# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["popuation_per_household"] = housing["population"]/housing["households"]
#############################################################
# separat between X and y
#############################################################
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
# X = imputer.fit_transform(housing_num)
# housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
#
#################################################################
# preprocessing
# Pipeline
# ColumnTransformer
#################################################################
from sklearn.preprocessing import OneHotEncoder
# housing_cat = housing[["ocean_proximity"]]
# cat_encoder = OneHotEncoder()
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
#print(cat_encoder.categories_)

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]/X[:, households_ix]
        population_per_household = X[:, population_ix]/X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)
####################################################################
# start training
####################################################################
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
#print(lin_reg.score(housing_prepared, housing_labels))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
joblib.dump(lin_reg, "modelC2/lin_reg.pkl")
#print(np.sqrt(lin_mse))

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions_tree = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions_tree)
joblib.dump(tree_reg, "modelC2/tree_reg.pkl")
#print(np.sqrt(tree_mse))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
scores_lin = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
lin_remse_scores = np.sqrt(-scores_lin)

#print(tree_rmse_scores.mean())
#print(tree_rmse_scores.std())

###################################################
# hyperparameter fine-tuning
###################################################

from sklearn.model_selection import GridSearchCV

param_grid = [
    {"n_estimators": [3,10,30], "max_features": [2,4,6,8]},
    {"bootstrap":[False], "n_estimators": [3,10], "max_features": [2,3,4]}
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
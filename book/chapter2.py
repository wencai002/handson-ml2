import os
import tarfile
import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

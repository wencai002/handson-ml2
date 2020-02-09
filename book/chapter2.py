import os
import tarfile
import urllib
import pandas as pd

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
print(housing.head())
print(housing.info())
print(housing.describe())

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(16,10))
plt.show()

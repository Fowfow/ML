#!/usr/bin/python3


## for data
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
from scipy.sparse import data
import seaborn as sns
## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
## for explainer
from lime import lime_tabular

## Data Analysis
from tools.utils import data_analysis

import warnings
warnings.filterwarnings("ignore")



################
# Import data
################
dtf = pd.read_csv('input_data/data_titanic.csv')
dtf.head()



data_analysis.dtf_overview(dtf)
dtf = dtf.set_index("PassengerId")
dtf = dtf.rename(columns={"Survived":"Y"})

features=[]
data_analysis.freqdist_plot(dtf, "Y", figsize=(5, 3))
data_analysis.corr_matrix(dtf, method="pearson", negative=False, lst_filters=["Y"], figsize=(15, 1))
pps = data_analysis.pps_matrix(dtf, lst_filters=["Y"], figsize=(15,1))
data_analysis.bivariate_plot(dtf, x="Sex", y="Y", figsize=(10, 5))
# coeff, p = data_analysis.test_corr(dtf, x="Sex", y="Y")
# data_analysis.nan_analysis(dtf, na_x="Age", y="Y", max_cat=20, figsize=(10,5))
# data_analysis.cross_distributions(dtf, "Age", "Y", "Sex")
# plt.show()







plt.show()
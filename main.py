#!/usr/bin/python3


## for data
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
# from scipy.sparse import data
import seaborn as sns
## for statistical tests
# import scipy
# import statsmodels.formula.api as smf
# import statsmodels.api as sm
import ppscore
## for machine learning
# from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
## for explainer
# from lime import lime_tabular

import warnings
warnings.filterwarnings("ignore")

## Data Analysis
from tools.utils import data_analysis, ml_utils
from input_data.generate_input import generate_input_csv



################
# Import data
################
dtf_tit = pd.read_csv('input_data/data_titanic.csv')
dtf_tit.head()
dtf_tit = dtf_tit.rename(columns={"Survived":"Y"})

# data_analysis.dtf_overview(dtf)
# dtf = dtf.set_index("PassengerId")
# dtf = dtf.rename(columns={"Survived":"Y"})

# features=[]
# data_analysis.freqdist_plot(dtf, "Y", figsize=(5, 3))
# data_analysis.corr_matrix(dtf, method="pearson", negative=False, lst_filters=["Y"], figsize=(15, 1))
# pps = data_analysis.pps_matrix(dtf, figsize=(15,1))
# matrix_df = data_analysis.pps_matrix(dtf)
# data_analysis.bivariate_plot(dtf, x="Age", y="Y", figsize=(10, 5))
# coeff, p = data_analysis.test_corr(dtf, x="Sex", y="Y")
# data_analysis.nan_analysis(dtf, na_x="Cabin", y="Survived", max_cat=20, figsize=(10,5))
# data_analysis.cross_distributions(dtf, "Age", "Y", "Sex")
# plt.show()

# df = pd.DataFrame()
# df["x"] = np.random.uniform(-2, 2, 1_000_000)
# df["error"] = np.random.uniform(-0.5, 0.5, 1_000_000)
# df["y"] = df["x"] * df["x"] + df["error"]


# print(ppscore.score(df, "x", "y"))
# predictors_df = ppscore.predictors(df, "y")
# sns.barplot(data=predictors_df, x="x", y="ppscore")
# matrix_df = ppscore.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
# sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)

# plt.show()
# ppscore.matrix(df)



############
# CCS
############
# generate_input_csv(filename="input_data/ccs.csv", nrows=100, ncols=4)
dtf_ccs = pd.read_csv("input_data/ccs.csv")
dtf_ccs.head()
dtf_ccs = dtf_ccs.rename(columns={"S":"Y"})

# data_analysis.dtf_overview(dtf)
# data_analysis.freqdist_plot(dtf, x="C1")
# data_analysis.bivariate_plot(dtf, "C1", "C2")
# data_analysis.nan_analysis(dtf, na_x="C1", y="C2")
# data_analysis.ts_analysis(dtf, "Id", "C2")
# data_analysis.cross_distributions(dtf, "C1", "C2", "S")
# data_analysis.corr_matrix(dtf)
# data_analysis.pps_matrix(dtf)
# data_analysis.test_corr(dtf, x="C1", y="S")
# data_analysis.dtf_partitioning(dtf, "S")
# data_analysis.test_corr(dtf, "C2", "S")
# train, test = data_analysis.dtf_partitioning(dtf, "S")
dtf_tit_bal = data_analysis.rebalance(dtf_tit, y="Y", balance="up")

print(dtf_tit_bal)
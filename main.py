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





################
# Import data
################
dtf = pd.read_csv('input_data/data_titanic.csv')
dtf.head()

# data_analysis.dtf_overview(dtf)
# data_analysis.freqdist_plot(dtf, "Embarked")
# data_analysis.bivariate_plot(dtf, x="Age", y="Fare")
# data_analysis.nan_analysis(dtf, "Sex", "Age")
# data_analysis.cross_distributions(dtf, "Age", "Survived", "Sex")
# data_analysis.corr_matrix(dtf)
data_analysis.pps_matrix(dtf)






plt.show()
#!/usr/bin/python3


## for data
from numpy.core.numeric import False_
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
# from scipy.sparse import data
# from scipy.sparse import data
# import seaborn as sns
## for statistical tests
import scipy
# import statsmodels.formula.api as smf
# import statsmodels.api as sm
# import ppscore
## for machine learning
# from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from sklearn import ensemble, model_selection, metrics, svm
## for deep learning
from tensorflow.keras import models, layers
## for explainer
# from lime import lime_tabular

import warnings
warnings.filterwarnings("ignore")

## Data Analysis
from tools.utils import data_analysis
from input_data.generate_input import generate_input_csv

################
# Import data
################
# df = pd.DataFrame()
# df["x"] = np.random.uniform(-2, 2, 1_000_000)
# df["error"] = np.random.uniform(-0.5, 0.5, 1_000_000)
# df["y"] = df["x"] * df["x"] + df["error"]
# CCS
# generate_input_csv(filename="input_data/ccs.csv", nrows=100, ncols=4)
# dtf_ccs = pd.read_csv("input_data/ccs.csv")
# dtf_ccs.head()
# dtf_ccs = dtf_ccs.rename(columns={"S":"Y"})
# dtf_ccs.loc[dtf_ccs.C1 < 0, "C1"] = np.nan
# print(dtf_ccs.head(100))

# Titanic
dtf_tit = pd.read_csv('input_data/data_titanic.csv')
# dtf_tit.head()
dtf_tit = dtf_tit.set_index("PassengerId")
dtf_tit = dtf_tit.rename(columns={"Survived":"Y"})

############
# Manual Feature Analysis
############
print('-------------------------')
print('Manual Feature analysis')
print('-------------------------')  
# data_analysis.dtf_overview(dtf)
# data_analysis.(dtf, method="pearson", negative=False, lst_filters=["Y"], figsize=(15, 1))
# pps = data_analysis.pps_matrix(dtf_tit, figsize=(15,1))
# data_analysis.bivariate_plot(dtf_tit, x="Age", y="Y", figsize=(10, 5))
# data_analysis.nan_analysis(dtf, na_x="Cabin", y="Survived", max_cat=20, figsize=(10,5))
# data_analysis.cross_distributions(dtf_tit, "Sex", "Age", "Pclass")
# plt.show()
# data_analysis.ts_analysis(dtf, "Id", "C2")
# data_analysis.freqdist_plot(dtf_tit, x="Cabin_section", top=10)
# data_analysis.cross_distributions(dtf_tit, x1="Cabin_section", x2="Pclass", y="Y", figsize=(10,5))
# coef, p = data_analysis.test_corr(dtf_tit, x="Cabin_section", y="Y")
# coeff, p = data_analysis.test_corr(dtf_tit, x="Sex", y="Y")
# corr = data_analysis.corr_matrix(dtf_tit, method="pearson", negative=False, lst_filters=["Y"], figsize=(15,1)) # USEFUL
# pps = data_analysis.pps_matrix(dtf_tit, lst_filters=["Y"], figsize=(15,1))  # USEFUL
# predictors_df = ppscore.predictors(dtf_tit, "Y")
# sns.barplot(data=predictors_df, x="x", y="ppscore")
# plt.show()
dtf_tit["Cabin_section"] = dtf_tit["Cabin"].apply(lambda x : str(x)[0]) # USEFUL
features = ["Sex", "Age", "Embarked", "Pclass", "Fare", "Cabin_section", "SibSp", "Parch"]
print("\t", "You selected the following features : ", features)
dtf_tit=dtf_tit[features+["Y"]]
############
# Data Preprocessing
############
print('-------------------------')
print('DTF preprocessing')
print('-------------------------')    
data_analysis.dtf_overview(dtf_tit, figsize = (15,15), filename="pre_overview.png")
ret = data_analysis.data_preprocessing(dtf=dtf_tit, y="Y", processNas="mean", processCategorical=["Sex", "Embarked", "Pclass", "Cabin_section"],
                                        split=None, scale="minmax", task="classification")
dtf_tit = ret["dtf"]
data_analysis.dtf_overview(dtf_tit, figsize = (15,15), filename="post_overview.png")
# Splitting
dtf_train, dtf_test = data_analysis.dtf_partitioning(dtf_tit, y="Y", test_size=0.3, shuffle=False)
dtf_train = data_analysis.pop_columns(dtf=dtf_train, lst_cols=["Y"], where="end")
# check = data_analysis.rebalance(dtf_train, y="Y", balance=None)
############
# Feature Selection
############
print('-------------------------')
print('Feature Selection')
print('-------------------------')
feature_selection_switch = True
X_names = []
if (feature_selection_switch):
    pps = data_analysis.pps_matrix(dtf_train, lst_filters = [], annotation=True, figsize=(25,15))
    dic_feat_sel = data_analysis.features_selection(dtf_train, y="Y", task="classification", top=10, figsize=(12,5))
    plt.show()
    # -- Feature importance ---#
    model = ensemble.RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0, n_jobs = 20)
    # --
    # Permutation feature importance overcomes limitations of the impurity-based feature importance: 
    # they do not have a bias toward high-cardinality features and can be computed on a left-out test set.
    # --
    feat_imp = data_analysis.features_importance(   X=dtf_train.drop("Y",axis=1).values, y=dtf_train["Y"].values, 
                                                    X_names=dtf_train.drop("Y",axis=1).columns.tolist(), 
                                                    model=model, task="classification", figsize=(15, 13))
    X_names = feat_imp["VARIABLE"].to_list()
else:
    X_names = ["Age", "Fare", "Sex_male", "SibSp", "Pclass_3", "Parch", "Cabin_section_n", "Embarked_S", "Pclass_2",
               "Cabin_section_F", "Cabin_section_E", "Cabin_section_D"]
print("\t", "Final selected features : ", X_names)           
X_train = dtf_train[X_names].values
y_train = dtf_train["Y"].values
############
# Model Design
############
print('-------------------------')
print('Model Design (hyperparameters tuning)')
print('-------------------------')
search_best_estimator_parameters_switch = False
model_XGB = ensemble.GradientBoostingClassifier() # see other options
model_SVC = svm.SVC()
if (search_best_estimator_parameters_switch):
    param_dic = {"XGB":
                    {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],     #weighting factor for the corrections by new trees when added to the model
                    'n_estimators':[100,250,500,750,1000,1250,1500,1750],  #number of trees added to the model
                    'max_depth':[2,3,4,5,6,7],                             #maximum depth of the tree
                    'min_samples_split':[2,4,6,8,10,20,40,60,100],         #sets the minimum number of samples to split
                    'min_samples_leaf':[1,3,5,7,9],                        #the minimum number of samples to form a leaf
                    'max_features':[2,3,4,5,6,7],                          #square root of features is usually a good starting point
                    'subsample': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]},    # the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.
                 "SVC":
                    {"kernel": ["rbf", "linear"],
                     "gamma": scipy.stats.expon(scale=.1),
                     "C": scipy.stats.expon(scale=100),
                     "class_weight": ["balanced", None]}
                 }  
    # Search for best estimator parameters
    # this takes a while
    model = data_analysis.tune_classif_model(X_train, y_train, model_SVC, param_dic["SVC"], scoring="accuracy", searchtype="RandomSearch", n_iter=1000, cv=10, figsize=(10,5))
else:
    # Instant best estimator parameters
    best_params_XGB = {'subsample': 0.85, 'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_features': 6, 'max_depth': 6, 'learning_rate': 0.05}
    best_param_SVC = {}
    model.set_params(**best_params_XGB)
    print("--- Kfold Validation ---")
    dic_scores = {  'accuracy':metrics.make_scorer(metrics.accuracy_score),
                    'precision':metrics.make_scorer(metrics.precision_score), 
                    'recall':metrics.make_scorer(metrics.recall_score),
                    'f1':metrics.make_scorer(metrics.f1_score)}
    Kfold_base = model_selection.cross_validate(estimator=model, X=X_train, y=y_train, cv=10, scoring=dic_scores, n_jobs=20)
    Kfold_model = model_selection.cross_validate(estimator=model, X=X_train, y=y_train, cv=10, scoring=dic_scores, n_jobs=20)
    for score in dic_scores.keys():
        print("\t", score, "mean - base model:", round(Kfold_base["test_"+score].mean(),2), " --> best model:", round(Kfold_model["test_"+score].mean(),2))
    kfold_roc_switch = True
    if (kfold_roc_switch): data_analysis.utils_kfold_roc(model, X_train, y_train, cv=10, figsize=(10,5))
    ## Threshold analysis
    print("--- Threshold Selection ---")
    select_threshold_switch = True
    if (select_threshold_switch): data_analysis.utils_threshold_selection(model, X_train, y_train, figsize=(10,5))
#-----------------
# Train/Test
#-----------------
print('-------------------------')
print('Fit (Train/Test) using the tuned classification model')
print('-------------------------')
X_test = dtf_test[X_names].values
y_test = dtf_test["Y"].values
model, predicted_prob, predicted = data_analysis.fit_ml_classif(model, X_train, y_train, X_test, threshold=0.5)
#-----------------
# Evaluate the tuned classification model
#-----------------
print('-------------------------')
print('Evaluating the tuned classification model')
print('-------------------------')
evaluate_switch = True
if (evaluate_switch): data_analysis.evaluate_classif_model(y_test, predicted, predicted_prob, figsize=(25,5))
#-----------------
# Explainability
#-----------------
explain_switch = True
if (explain_switch):
    i = 3
    print("True:", y_test[i], "--> Pred:", int(predicted[i]), "| Prob:", np.round(np.max(predicted_prob[i]), 2))
    data_analysis.explainer_shap(model, X_names, X_instance=X_test[i], X_train=None, task="classification", top=10)
    data_analysis.explainer_lime(X_train, X_names, model, y_train, X_test[i], task="classification", top=10)
    plt.show()
#-----------------
# Visualization
#-----------------
print('-------------------------')
print('Visualising the tuned classification model')
print('-------------------------')
visual_switch = False
if (visual_switch):
    model2d = ensemble.GradientBoostingClassifier()
    model2d.set_params(**{'subsample':1, 'n_estimators':1750, 'min_samples_split':6, 'min_samples_leaf':1, 'max_depth':4, 'learning_rate':0.001})
    data_analysis.plot2d_classif_model(X_train, y_train, X_test, y_test, model2d, annotate=False, figsize=(10,5))


###############################
# Deep learning
###############################
#-----
# Feature selection
#-----
X_names = dtf_train.drop("Y", axis=1).columns.tolist()
X_train = dtf_train.drop("Y", axis=1).values
y_train = dtf_train["Y"].values
X_test = dtf_test[X_names].values
#----
# Model Design
#----
print('-------------------------')
print('Model Design (hyperparameters tuning)')
print('-------------------------')
### build ann
# n_features = X_train.shape[1]
# n_neurons = int(round((n_features + 1)/2))
# switch = False_
# model_dl = models.Sequential([
#                             layers.Dense(input_dim=n_features, units=n_neurons,
#                                          kernel_initializer='uniform', activation='relu'),
#                             layers.Dropout(rate=0.2),
#                             layers.Dense(units=n_neurons,
#                                          kernel_initializer='uniform', activation='relu'),
#                             layers.Dropout(rate=0.2),
#                             layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
#                             ])
# if (switch):
#     param_dic = {"SEQ":
#                     {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],     #weighting factor for the corrections by new trees when added to the model
#                     'n_estimators':[100,250,500,750,1000,1250,1500,1750],  #number of trees added to the model
#                     'max_depth':[2,3,4,5,6,7],                             #maximum depth of the tree
#                     'min_samples_split':[2,4,6,8,10,20,40,60,100],         #sets the minimum number of samples to split
#                     'min_samples_leaf':[1,3,5,7,9],                        #the minimum number of samples to form a leaf
#                     'max_features':[2,3,4,5,6,7],                          #square root of features is usually a good starting point
#                     'subsample': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]}     # the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.
#                  }  
#     # Search for best estimator parameters
#     # this takes a while
#     model_dl = data_analysis.tune_classif_model(X_train, y_train, model_SVC, param_dic["SEQ"], scoring="accuracy", searchtype="RandomSearch", n_iter=1000, cv=10, figsize=(10,5))    
# else:
#     # Instant best estimator parameters
#     best_params = {'subsample': 0.85, 'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_features': 6, 'max_depth': 6, 'learning_rate': 0.05}
#     model_dl.set_params(**best_params)
#     print("--- Kfold Validation ---")
#     dic_scores = {  'accuracy':metrics.make_scorer(metrics.accuracy_score),
#                     'precision':metrics.make_scorer(metrics.precision_score), 
#                     'recall':metrics.make_scorer(metrics.recall_score),
#                     'f1':metrics.make_scorer(metrics.f1_score)}
#     Kfold_base = model_selection.cross_validate(estimator=model_dl, X=X_train, y=y_train, cv=10, scoring=dic_scores, n_jobs=20)
#     Kfold_model = model_selection.cross_validate(estimator=model_dl, X=X_train, y=y_train, cv=10, scoring=dic_scores, n_jobs=20)
#     for score in dic_scores.keys():
#         print("\t", score, "mean - base model:", round(Kfold_base["test_"+score].mean(),2), " --> best model:", round(Kfold_model["test_"+score].mean(),2))
#     kfold_roc_switch = False
#     if (kfold_roc_switch): data_analysis.utils_kfold_roc(model_dl, X_train, y_train, cv=10, figsize=(10,5))
#     ## Threshold analysis
#     print("--- Threshold Selection ---")
#     select_threshold_switch = False
#     if (select_threshold_switch): data_analysis.utils_threshold_selection(model_dl, X_train, y_train, figsize=(10,5))
#----
# Train/test
#----
model_dl, predicted_prob, predicted = data_analysis.fit_dl_classif(X_train, y_train, X_test, model=None, batch_size=32, epochs=100, threshold=0.5)
#--
# Visualize
#--
data_analysis.visualize_nn(model_dl)









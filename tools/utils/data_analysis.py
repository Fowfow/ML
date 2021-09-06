
## for data
import numpy as np
import pandas as pd

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
import ppscore

## for machine learning
from sklearn import preprocessing, impute, utils, linear_model, feature_selection, model_selection, metrics, decomposition, cluster, ensemble
import imblearn

## for deep learning
from tensorflow.keras import models, layers
import minisom

## for explainer
from lime import lime_tabular
import shap

## for geospatial
import folium
import geopy


###############################################################################
#                       DATA ANALYSIS                                         #
###############################################################################

def utils_recognize_type(dtf, col, max_cat=20):
    """
    Recognize whether a column is numerical or categorical.
    :parameter
        :param dtf: dataframe - input data
        :param col: str - name of the column to analyze
        :param max_cat: num - max number of unique values to consider a column as categorical
    :return
        "cat" if the column is categorical or "num" otherwise
    """

    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"


def dtf_overview(dtf, max_cat=20, figsize=(10,5)):
    '''
    Get a general overview of a dataframe.
    :parameter
        :param dtf: dataframe - input data
        :param max_cat: num - mininum number of recognize column type
    '''    
    ## recognize column type
    dic_cols = {col:utils_recognize_type(dtf, col, max_cat=max_cat) for col in dtf.columns}
        
    ## print info
    len_dtf = len(dtf)
    print("Shape:", dtf.shape)
    print("-----------------")
    for col in dtf.columns:
        info = col+" --> Type:"+dic_cols[col]
        info = info+" | Nas: "+str(dtf[col].isna().sum())+"("+str(int(dtf[col].isna().mean()*100))+"%)"
        if dic_cols[col] == "cat":
            info = info+" | Categories: "+str(dtf[col].nunique())
        else:
            info = info+" | Min-Max: "+"({x})-({y})".format(x=str(int(dtf[col].min())), y=str(int(dtf[col].max())))
        if dtf[col].nunique() == len_dtf:
            info = info+" | Possible PK"
        print(info)
                
    ## plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = dtf.isnull()
    for k,v in dic_cols.items():
        if v == "num":
            heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
        else:
            heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
    sns.heatmap(heatmap, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Dataset Overview')
    #plt.setp(plt.xticks()[1], rotation=0)
    plt.show()
    
    ## add legend
    print("\033[1;37;40m Categerocial \033[m", "\033[1;30;41m Numerical \033[m", "\033[1;30;47m NaN \033[m")


def check_pk(dtf, pk):
    '''
    Check the primary key of a dtf
    :parameter
        :param dtf: dataframe - input data
        :param pk: str - column name
    '''
    unique_pk, len_dtf = dtf[pk].nunique(), len(dtf)
    check = "unique "+pk+": "+str(unique_pk)+"  |  len dtf: "+str(len_dtf)
    if unique_pk == len_dtf:
        msg = "OK!!!  "+check
        print(msg)
    else:
        msg = "WARNING!!!  "+check
        ERROR = dtf.groupby(pk).size().reset_index(name="count").sort_values(by="count", ascending=False)
        print(msg)
        print("Example: ", pk, "==", ERROR.iloc[0,0])


def pop_columns(dtf, lst_cols, where="front"):
    '''
    Moves columns into a dtf.
    :parameter
        :param dtf: dataframe - input data
        :param lst_cols: list - names of the columns that must be moved
        :param where: str - "front" or "end"
    :return
        dtf with moved columns
    '''    
    current_cols = dtf.columns.tolist()
    for col in lst_cols:    
        current_cols.pop( current_cols.index(col) )
    if where == "front":
        dtf = dtf[lst_cols + current_cols]
    elif where == "end":
        dtf = dtf[current_cols + lst_cols]
    return dtf


def freqdist_plot(dtf, x, max_cat=20, top=None, show_perc=True, bins=100, quantile_breaks=(0,10), box_logscale=False, figsize=(10,5)):
    '''
    Plots the frequency distribution of a dtf column.
    :parameter
        :param dtf: dataframe - input data
        :param x: str - column name
        :param max_cat: num - max number of uniques to consider a numerical variable as categorical
        :param top: num - plot setting
        :param show_perc: logic - plot setting
        :param bins: num - plot setting
        :param quantile_breaks: tuple - plot distribution between these quantiles (to exclude outilers)
        :param box_logscale: logic
        :param figsize: tuple - plot settings
    '''    
    try:
        ## cat --> freq
        if utils_recognize_type(dtf, x, max_cat) == "cat":   
            ax = dtf[x].value_counts().head(top).sort_values().plot(kind="barh", figsize=figsize)
            totals = []
            for i in ax.patches:
                totals.append(i.get_width())
            if show_perc == False:
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(i.get_width()), fontsize=10, color='black')
            else:
                total = sum(totals)
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=10, color='black')
            ax.grid(axis="x")
            plt.suptitle(x, fontsize=20)
            plt.show()
            
        ## num --> density
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x, fontsize=20)
            ### distribution
            ax[0].title.set_text('distribution')
            variable = dtf[x].fillna(dtf[x].mean())
            breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
            variable = variable[ (variable > breaks[quantile_breaks[0]]) & (variable < breaks[quantile_breaks[1]]) ]
            sns.distplot(variable, hist=True, kde=True, kde_kws={"shade":True}, ax=ax[0])
            des = dtf[x].describe()
            ax[0].axvline(des["25%"], ls='--')
            ax[0].axvline(des["mean"], ls='--')
            ax[0].axvline(des["75%"], ls='--')
            ax[0].grid(True)
            des = round(des, 2).apply(lambda x: str(x))
            box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
            ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=1))
            ### boxplot 
            if box_logscale == True:
                ax[1].title.set_text('outliers (log scale)')
                tmp_dtf = pd.DataFrame(dtf[x])
                tmp_dtf[x] = np.log(tmp_dtf[x])
                tmp_dtf.boxplot(column=x, ax=ax[1])
            else:
                ax[1].title.set_text('outliers')
                dtf.boxplot(column=x, ax=ax[1])
            plt.show()   
        
    except Exception as e:
        print("--- got error ---")
        print(e)


def bivariate_plot(dtf, x, y, max_cat=20, figsize=(10,5)):
    '''
    Plots a bivariate analysis.
    :parameter
        :param dtf: dataframe - input data
        :param x: str - column
        :param y: str - column
        :param max_cat: num - max number of uniques to consider a numerical variable as categorical
    '''
    try:
        ## num vs num --> stacked + scatter with density
        if (utils_recognize_type(dtf, x, max_cat) == "num") & (utils_recognize_type(dtf, y, max_cat) == "num"):
            ### stacked
            dtf_noNan = dtf[dtf[x].notnull()]  #can't have nan
            breaks = np.quantile(dtf_noNan[x], q=np.linspace(0, 1, 11))
            groups = dtf_noNan.groupby([pd.cut(dtf_noNan[x], bins=breaks, duplicates='drop')])[y].agg(['mean','median','size'])
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(x+"   vs   "+y, fontsize=20)
            groups[["mean", "median"]].plot(kind="line", ax=ax)
            groups["size"].plot(kind="bar", ax=ax, rot=45, secondary_y=True, color="grey", alpha=0.3, grid=True)
            ax.set(ylabel=y)
            ax.right_ax.set_ylabel("Observazions in each bin")
            plt.show()
            ### joint plot
            sns.jointplot(x=x, y=y, data=dtf, dropna=True, kind='reg', height=int((figsize[0]+figsize[1])/2) )
            plt.show()

        ## cat vs cat --> hist count + hist %
        elif (utils_recognize_type(dtf, x, max_cat) == "cat") & (utils_recognize_type(dtf, y, max_cat) == "cat"):  
            fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x+"   vs   "+y, fontsize=20)
            ### count
            ax[0].title.set_text('count')
            order = dtf.groupby(x)[y].count().index.tolist()     
            a = dtf.groupby(x)[y].count().reset_index()
            a = a.rename(columns={y:"tot"})    
            aa = dtf.groupby([x,y])[y].count()   
            aa = aa.reset_index(name="count")                 
            sns.barplot(x=x, y="count", hue=y, data=aa, ax=ax[0], order=order)
            ax[0].grid(True)
            ### percentage
            ax[1].title.set_text('percentage')
            b = dtf.groupby([x,y])[y].count()
            b = b.reset_index(name="count")
            b = b.merge(a, how="left")
            b["%"] = b["count"] / b["tot"] *100
            sns.barplot(x=x, y="%", hue=y, data=b, ax=ax[1], order= order).get_legend().remove()
            ax[1].grid(True)
            ### fix figure
            # plt.close(2)
            # plt.close(3)
            plt.show()
    
        ## num vs cat --> density + stacked + boxplot 
        else:
            if (utils_recognize_type(dtf, x, max_cat) == "cat"):
                cat,num = x,y
            else:
                cat,num = y,x
            fig, ax = plt.subplots(nrows=1, ncols=3,  sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x+"   vs   "+y, fontsize=20)
            ### distribution
            ax[0].title.set_text('density')
            for i in sorted(dtf[cat].unique()):
                sns.distplot(dtf[dtf[cat]==i][num], hist=False, label=i, ax=ax[0])
            ax[0].grid(True)
            ### stacked
            dtf_noNan = dtf[dtf[num].notnull()]  #can't have nan
            ax[1].title.set_text('bins')
            breaks = np.quantile(dtf_noNan[num], q=np.linspace(0,1,11))
            tmp = dtf_noNan.groupby([cat, pd.cut(dtf_noNan[num], breaks, duplicates='drop')]).size().unstack().T
            tmp = tmp[dtf_noNan[cat].unique()]
            tmp["tot"] = tmp.sum(axis=1)
            for col in tmp.drop("tot", axis=1).columns:
                tmp[col] = tmp[col] / tmp["tot"]
            tmp.drop("tot", axis=1)[sorted(dtf[cat].unique())].plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)
            ### boxplot   
            ax[2].title.set_text('outliers')
            sns.boxplot(x=cat, y=num, data=dtf, ax=ax[2], order=sorted(dtf[cat].unique()))
            ax[2].grid(True)
            ### fix figure
            plt.close(2)
            plt.close(3)
            plt.show()
        
    except Exception as e:
        print("--- got error ---")
        print(e)        


def nan_analysis(dtf, na_x, y, max_cat=20, figsize=(10,5)):
    '''
    Plots a bivariate analysis using Nan and not-Nan as categories.
    '''    
    dtf_NA = dtf[[na_x, y]]
    dtf_NA[na_x] = dtf[na_x].apply(lambda x: "Value" if not pd.isna(x) else "NA")
    bivariate_plot(dtf_NA, x=na_x, y=y, max_cat=max_cat, figsize=figsize)


def ts_analysis(dtf, x, y, max_cat=20, figsize=(10,5)):
    '''
    Plots a bivariate analysis with time variable.
    '''    
    if utils_recognize_type(dtf, y, max_cat) == "cat":
        dtf_tmp = dtf.groupby(x)[y].sum()       
    else:
        dtf_tmp = dtf.groupby(x)[y].median()
    dtf_tmp.plot(title=y+" by "+x, figsize=figsize, grid=True)
    plt.show()


def cross_distributions(dtf, x1, x2, y, max_cat=20, figsize=(10,5)):
    '''
    plots multivariate analysis.
    '''    
    ## Y cat
    if utils_recognize_type(dtf, y, max_cat) == "cat":
        
        ### cat vs cat --> contingency table
        if (utils_recognize_type(dtf, x1, max_cat) == "cat") & (utils_recognize_type(dtf, x2, max_cat) == "cat"):
            cont_table = pd.crosstab(index=dtf[x1], columns=dtf[x2], values=dtf[y], aggfunc="sum")
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(cont_table, annot=True, fmt='.0f', cmap="YlGnBu", ax=ax, linewidths=.5).set_title(x1+'  vs  '+x2+'  (filter: '+y+')')
    
        ### num vs num --> scatter with hue
        elif (utils_recognize_type(dtf, x1, max_cat) == "num") & (utils_recognize_type(dtf, x2, max_cat) == "num"):
            sns.lmplot(x=x1, y=x2, data=dtf, hue=y, height=figsize[1])
        
        ### num vs cat --> boxplot with hue
        else:
            if (utils_recognize_type(dtf, x1, max_cat) == "cat"):
                cat,num = x1,x2
            else:
                cat,num = x2,x1
            fig, ax = plt.subplots(figsize=figsize)
            sns.boxplot(x=cat, y=num, hue=y, data=dtf, ax=ax).set_title(x1+'  vs  '+x2+'  (filter: '+y+')')
            ax.grid(True)
        plt.show()
    ## Y num
    else:
        ### all num --> 3D scatter plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
        plot3d = ax.scatter(xs=dtf[x1], ys=dtf[x2], zs=dtf[y], c=dtf[y], cmap='inferno', linewidth=0.5)
        fig.colorbar(plot3d, shrink=0.5, aspect=5, label=y)
        ax.set(xlabel=x1, ylabel=x2, zlabel=y)
        plt.show()
     

###############################################################################
#                         CORRELATION                                         #
###############################################################################  

def corr_matrix(dtf, method="pearson", negative=True, lst_filters=[], annotation=True, figsize=(10,5)):    
    '''
    Computes the correlation matrix.
    :parameter
        :param dtf: dataframe - input data
        :param method: str - "pearson" (numeric), "spearman" (categorical), "kendall"
        :param negative: bool - if False it takes the absolute values of correlation
        :param lst_filters: list - filter rows to show
        :param annotation: logic - plot setting
    '''    
    ## factorize
    dtf_corr = dtf.copy()
    for col in dtf_corr.columns:
        if dtf_corr[col].dtype == "O":
            print("--- WARNING: Factorizing", dtf_corr[col].nunique(),"labels of", col, "---")
            dtf_corr[col] = dtf_corr[col].factorize(sort=True)[0]
    ## corr matrix
    dtf_corr = dtf_corr.corr(method=method) if len(lst_filters) == 0 else dtf_corr.corr(method=method).loc[lst_filters]
    dtf_corr = dtf_corr if negative is True else dtf_corr.abs()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dtf_corr, annot=annotation, fmt='.2f', cmap="YlGnBu", ax=ax, cbar=True, linewidths=0.5)
    plt.title(method + " correlation")
    plt.show()
    return dtf_corr


def pps_matrix(dtf, annotation=True, lst_filters=[], figsize=(10,5)):
    '''
    Computes the pps matrix.
    :parameter
        :param dtf: dataframe - feature matrix dtf
    '''    
    dtf_pps = ppscore.matrix(dtf) if len(lst_filters) == 0 else ppscore.matrix(dtf).loc[lst_filters]
    dtf_pps = dtf_pps[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dtf_pps, vmin=0., vmax=1., annot=annotation, fmt='.2f', cmap="YlGnBu", ax=ax, cbar=True, linewidths=0.5)
    # sns.heatmap(dtf_pps, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
    plt.title("predictive power score")
    plt.show()
    return dtf_pps


def test_corr(dtf, x, y, max_cat=20):
    '''
    Computes correlation/dependancy and p-value (prob of happening something different than what observed in the sample)
    '''
    ## num vs num --> pearson
    if (utils_recognize_type(dtf, x, max_cat) == "num") & (utils_recognize_type(dtf, y, max_cat) == "num"):
        dtf_noNan = dtf[dtf[x].notnull()]  #can't have nan
        coeff, p = scipy.stats.pearsonr(dtf_noNan[x], dtf_noNan[y])
        coeff, p = round(coeff, 3), round(p, 3)
        conclusion = "Significant" if p < 0.05 else "Non-Significant"
        print("Pearson Correlation:", coeff, conclusion, "(p-value: "+str(p)+")")
    
    ## cat vs cat --> cramer (chiquadro)
    elif (utils_recognize_type(dtf, x, max_cat) == "cat") & (utils_recognize_type(dtf, y, max_cat) == "cat"):
        cont_table = pd.crosstab(index=dtf[x], columns=dtf[y])
        chi2_test = scipy.stats.chi2_contingency(cont_table)
        chi2, p = chi2_test[0], chi2_test[1]
        n = cont_table.sum().sum()
        phi2 = chi2/n
        r,k = cont_table.shape
        phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
        rcorr = r-((r-1)**2)/(n-1)
        kcorr = k-((k-1)**2)/(n-1)
        coeff = np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))
        coeff, p = round(coeff, 3), round(p, 3)
        conclusion = "Significant" if p < 0.05 else "Non-Significant"
        print("Cramer Correlation:", coeff, conclusion, "(p-value: "+str(p)+")")
    
    ## num vs cat --> 1way anova (f: the means of the groups are different)
    else:
        if (utils_recognize_type(dtf, x, max_cat) == "cat"):
            cat,num = x,y
        else:
            cat,num = y,x
        model = smf.ols(num+' ~ '+cat, data=dtf).fit()
        table = sm.stats.anova_lm(model)
        p = table["PR(>F)"][0]
        coeff, p = None, round(p, 3)
        conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
        print("Anova F: the variables are", conclusion, "(p-value: "+str(p)+")")
        
    return coeff, p


###############################################################################
#                       PREPROCESSING                                         #
###############################################################################

def dtf_partitioning(dtf, y, test_size=0.3, shuffle=False):
    '''
    Split the dataframe into train / test
    '''
    dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=test_size, shuffle=shuffle) 
    print("X_train shape:", dtf_train.drop(y, axis=1).shape, "| X_test shape:", dtf_test.drop(y, axis=1).shape)
    print("y_train mean:", round(np.mean(dtf_train[y]),2), "| y_test mean:", round(np.mean(dtf_test[y]),2))
    print(dtf_train.shape[1], "features:", dtf_train.drop(y, axis=1).columns.to_list())
    return dtf_train, dtf_test


def rebalance(dtf, y, balance=None,  method="random", replace=True, size=1):
    '''
    Rebalances a dataset with up-sampling and down-sampling.
    :parameter
        :param dtf: dataframe - feature matrix dtf
        :param y: str - column to use as target 
        :param balance: str - "up", "down", if None just prints some stats
        :param method: str - "random" for sklearn or "knn" for imblearn
        :param size: num - 1 for same size of the other class, 0.5 for half of the other class
    :return
        rebalanced dtf
    '''
    ## check
    print("--- situation ---")
    check = dtf[y].value_counts().to_frame()
    check["%"] = (check[y] / check[y].sum() *100).round(1).astype(str) + '%'
    print(check)
    print("tot:", check[y].sum())

    ## sklearn
    if balance is not None and method == "random":
        ### set the major and minor class
        major = check.index[0]
        minor = check.index[1]
        dtf_major = dtf[dtf[y]==major]
        dtf_minor = dtf[dtf[y]==minor]

        ### up-sampling
        if balance == "up":
            print("--- upsampling ---")
            print("   randomly replicate observations from the minority class (Overfitting risk)")
            dtf_minor = utils.resample(dtf_minor, replace=replace, random_state=123, n_samples=int(size*len(dtf_major)))
            dtf_balanced = pd.concat([dtf_major, dtf_minor])

        ### down-sampling
        elif balance == "down":
            print("--- downsampling ---")
            print("   randomly remove observations of the majority class (Underfitting risk)")
            dtf_minor = utils.resample(dtf_minor, replace=replace, random_state=123, n_samples=int(size*len(dtf_major)))
            dtf_balanced = pd.concat([dtf_major, dtf_minor])

    ## imblearn
    if balance is not None and method == "knn":
        ### up-sampling
        if balance == "up":
            print("--- upsampling ---")
            print("   create synthetic observations from the minority class (Distortion risk)")
            smote = imblearn.over_sampling.SMOTE(random_state=123)
            dtf_balanced, y_values = smote.fit_sample(dtf.drop(y,axis=1), y=dtf[y])
            dtf_balanced[y] = y_values
       
        ### down-sampling
        elif balance == "down":
            print("--- downsampling ---")
            print("   select observations that don't affect performance (Underfitting risk)")
            nn = imblearn.under_sampling.CondensedNearestNeighbour(random_state=123)
            dtf_balanced, y_values = nn.fit_sample(dtf.drop(y,axis=1), y=dtf[y])
            dtf_balanced[y] = y_values
        
    ## check rebalance
    if balance is not None:
        print("--- new situation ---")
        check = dtf_balanced[y].value_counts().to_frame()
        check["%"] = (check[y] / check[y].sum() *100).round(1).astype(str) + '%'
        print(check)
        print("tot:", check[y].sum())
        return dtf_balanced
    


def fill_na(dtf, x, value=None):
    '''
    Replace Na with a specific value or mean for numerical and mode for categorical. 
    '''
    if value is None:
        value = dtf[x].mean() if utils_recognize_type(dtf, x) == "num" else dtf[x].mode().iloc[0]
        print("--- Replacing Nas with:", value, "---")
        dtf[x] = dtf[x].fillna(value)
        return dtf, value
    else:
        print("--- Replacing Nas with:", value, "---")
        dtf[x] = dtf[x].fillna(value)
        return dtf











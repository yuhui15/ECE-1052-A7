"""
Fill in the missing code. The lines with missing code have the string "#####" or '*'
"INSTRUCTIONS" comments explain how to fill in the mising code.
the outputfile.txt has the printouts from the program.
Your results should be similar within reason, if not, re-run the program,
since we are using RandomizedSearchCV, meaning there is some randomness involved.
Actually, we added np.random.seed() to fix the results, so you can check them.

You will be filling in code in two types of models:
1. a regression model and
2. a classification model.

Most of the time, because of similarities,
you can cut and paste from one model to the other.
But in a few instances, you cannot do this, so
you need to pay attention.
Also, in some cases,
you will find a "hint" for a solution 
in one of the two scripts (regression or classification)
that you can use as inspiration for the other.

This double task gives you the opportunity to look at the results
in both regression and classification approaches.

At the bottom, you will find some questions that we pose.
You do not need to write and turn in the answer to these questions,
but we strongly recommend you find out the answers to them.

"""
"""
In this homework you will implement a multi-class classifier.
You will also practice optimizing Ta-lib parameters further using ColumnTransformer and FunctionTransformer.
This is important because Ta-lib default parameter values do not work for all stocks or 
for all frequency samplings (mostly they work mostly for daily sampling and very liquid stocks).
The Ta-lib functions you will be using are RSI, ADX and SAR.
For information on these Ta-lib functions read: RSI.ADX.SAR.pptx
These are momentum indicators that require the preservation of the chronological order of the data.

Why not just binary target features?
Using a ternary target feature allows the model to pay attention to targets with large movements,
and pay less attention to targets with average movements.

To understand the scoring for multi-class classifiers you need to read ModelEvaluation.pdf
ModelEvaluation.pdf explains the various scorings used to evaluate classifiers but
concentrate on just two: accuracy and the F1 score.


Scorers:
Accuracy is a favorite scorer for classifiers, but it requires the target classes to be balanced.
You correct the lack of balance using weights.
Caution: using weights can cause a model fo overfit.

accuracy: Balanced labels only. Use with class_weight='balanced'
phi_k: Balanced or unbalanced labels. No need for class_weight='balanced'
phi (matthews_corrcoef): Balanced or unbalanced. No need for class_weight='balanced'
f1: Balanced or unbalanced. No need for class_weight='balanced'
f1_macro: All label categories are equally important no matter the number of samples belonging to each class. No need for class_weight="balanced"
f1_weighted: Label categories are weighted by the number of samples belonging to each class. No need for class_weight="balanced"


"""
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import fAux
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import talib as ta

np.random.seed(1) #to fix the results
rs = 2
ANNUALIZATION = 252 * 8 #8 H3 bars per trading day (was 252, which treated each 3-hour bar as a day)
 
file_path = 'outputfile.txt'
sys.stdout = open(file_path, "w")

df = pd.read_csv('USDCAD_H3_200001030000_202107201800.csv', sep='\t')

df['<DATETIME>'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df = df.set_index('<DATETIME>')
df.drop(['<TIME>'], axis=1, inplace=True)
df.drop(['<DATE>'], axis=1, inplace=True)

#save the open for white reality check
openp = df['<OPEN>'].copy()
close = df['<CLOSE>'].copy()

#buld the best window features after the exploratory data analysis:
for n in list(range(1,15)):
    name = 'ret' + str(n)
    df[name] = df["<OPEN>"].pct_change(periods=n) #for trading with open
    

#build date-time features
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values

#build target assuming we know today's open
df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1).fillna(0) #if you enter the trade immediately after the open


"""
INSTRUCTIONS
Using the example of the 2-label target as inspiration
construct a 3 label target called retFut1_categ
where the returns are binned into 3 equal quartiles: 
0.34 and below, getting label -1
0.34 to 0.66,  getting label 0
0.66 and above, getting label 1
"""
#transform the target
#you have the option of a 3 label target or a 2 label target

#2-label target
#df['retFut1_categ']=0
#df.loc[df['retFut1']>df['retFut1'].iloc[:10000].quantile(q=0.51),'retFut1_categ']=1

#3-label target gets better results
df['retFut1_categ']=0
df.loc[df['retFut1']>df['retFut1'].iloc[:10000].quantile(q=0.66),'retFut1_categ']=1
df.loc[df['retFut1']<df['retFut1'].iloc[:10000].quantile(q=0.34),'retFut1_categ']=-1


"""
INSTRUCTIONS
Use df.groupby to find out if the target labels (0, 1, and -1) are balanced
if they are not, you need to choose an appropriate metric e.g. phik or f1_macro 
if you want to use accuracy, you need to use class_weight='balanced' in the estimator
to make the importance of the lables match their frequency
"""

#always check to see that the labels occur in equal numbers
dfgroups = df.groupby("retFut1_categ").count() #if they do not, use "balanced" parameter in the estimator (see below)


#Since we are trading right after the open, 
#we only know yesterday's  high low close volume spread etc.
df['<HIGH>'] = df['<HIGH>'].shift(1)
df['<LOW>'] = df['<LOW>'].shift(1)
df['<CLOSE>'] = df['<CLOSE>'].shift(1)
df['<VOL>'] = df['<VOL>'].shift(1)
df['<SPREAD>'] = df['<SPREAD>'].shift(1)

#Build some Ta-lib features, with n=10, to be optimized in the pipeline
n=10
df['RSI']=ta.RSI(np.array(df['<CLOSE>']), timeperiod=n)
df['SMA'] = df['<CLOSE>'].rolling(window=n).mean()
df['Corr']= df['<CLOSE>'].rolling(window=n).corr(df['SMA'])
df['SAR']=ta.SAR(np.array(df['<HIGH>']),np.array(df['<LOW>']), 0.2,0.2)
df['ADX']=ta.ADX(np.array(df['<HIGH>']),np.array(df['<LOW>']), np.array(df['<OPEN>']), timeperiod =n)
df['OO']= df['<OPEN>']-df['<OPEN>'].shift(1)
df['OC']= df['<OPEN>']-df['<CLOSE>']
#Level-free replacements for the price-level features (<OPEN>, <HIGH>, <LOW>, <CLOSE>, SMA, SAR are not model
#features, see below). Built from the already-shifted HIGH/LOW/CLOSE (and SMA/SAR computed from them), so no look-ahead.
df['RANGE'] = (df['<HIGH>'] - df['<LOW>']) / df['<OPEN>'] #previous bar's range (volatility)
df['DSMA'] = (df['<OPEN>'] - df['SMA']) / df['<OPEN>'] #distance of the open from the 10-bar moving average
df['DSAR'] = (df['<OPEN>'] - df['SAR']) / df['<OPEN>'] #distance of the open from the parabolic SAR
#VOL: rolling volatility = std of open-to-open returns over the past 500 bars (about 2.5 months); uses opens up to
#the current bar only, so no look-ahead. It replaces the price levels as the volatility-regime feature (see below).
df['VOL'] = df['<OPEN>'].pct_change(1).rolling(500, min_periods=50).std()
df.fillna(0, inplace=True)

#select the features (by dropping)
cols_to_drop = ["<TICKVOL>","<VOL>","<SPREAD>"]  #optional
df_filtered = df.drop(cols_to_drop, axis=1)

#distribute the df data into X inputs and y target
X = df_filtered.drop(['retFut1', 'retFut1_categ'], axis=1) 
y = df_filtered[['retFut1_categ']]

#select the samples
x_train = X.iloc[0:10000]
x_test = X.iloc[10000:12000]

y_train = y.iloc[0:10000]
y_test = y.iloc[10000:12000]

df_train = df_filtered.iloc[0:10000]
df_test = df_filtered.iloc[10000:12000]


##########################################################################################################################

#set up the grid search and fit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer 
import phik
from phik.report import plot_correlation_matrix
from scipy.special import ndtr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import matthews_corrcoef
from sklearn.svm import LinearSVC
import detrendPrice 
import WhiteRealityCheckFor1 


def phi_k(y_true, y_pred):
    dfc = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    try:
        phi_k_corr = dfc.phik_matrix(interval_cols=[], njobs=1).iloc[1,0]
        phi_k_sig  = dfc.significance_matrix(interval_cols=[], njobs=1).iloc[1,0]
        phi_k_p_val = 1 - ndtr(phi_k_sig) 
    except:
        phi_k_corr = 0
        phi_k_p_val = 0
    #print(phi_k_corr)
    print(phi_k_p_val)
    return phi_k_corr

#ph_k is supposed to be better than phi, provided here for comparison
def phi(y_true, y_pred):
    mcc = matthews_corrcoef(y_true,y_pred) #a.k.a. phi
    print (mcc)
    return mcc


"""
INSTRUCTIONS
Use grid search to optimize the value og the talib feature timeperiod parameter.
Why not use the default value?
Because that default value is optimal for stocks, not currencies.

add two lines code that calculate
x['RSI'] and x['ADX'] using the incoming timeperiod parameter
that is selected from a parameter grid
use previous definition above of RSI and ADX


make sure  RSI_ADX_optimizer is working properly
the way it is programmed, it assumes that x is coming in as a dataframe.
"""

#for optimizing the timeperiod of RSI and ADX
#x is a dataframe because this happens first in the pipeline
#(the 'talib' ColumnTransformer below is set to pandas output, so x arrives as a DataFrame
#with its original column names and the result stays a DataFrame for the next step)
def RSI_ADX_optimizer(x, timeperiod=None):
    x = x.copy() #do not modify the caller's (sliced) DataFrame in place
    if timeperiod is None: #ta-lib needs an int (None happens when kw_args is not set)
        timeperiod = 14 #ta-lib's default for RSI and ADX
    x['RSI'] = ta.RSI(x['<CLOSE>'].to_numpy(dtype=np.float64), timeperiod=timeperiod)
    x['ADX'] = ta.ADX(x['<HIGH>'].to_numpy(dtype=np.float64), x['<LOW>'].to_numpy(dtype=np.float64), x['<OPEN>'].to_numpy(dtype=np.float64), timeperiod=timeperiod)
    return x

#myscorer= "accuracy"  #same as None; 
#myscorer = make_scorer(phi_k, greater_is_better=True)
myscorer="f1_macro"
#my_scorer="f1_weighted" 
#myscorer = make_scorer(phi, greater_is_better=True)

#when using smoother, use TimesSeriesSplit
#split = 5 
#split = TimeSeriesSplit(n_splits=5, max_train_size=2000) #fixed size window
split = TimeSeriesSplit(n_splits=5)

"""
INSTRUCTIONS
Use a FunctionTransformer 
to wrap RSI_ADX_optimizer into an object that has the fit_transform method.
For guidance see: https://archive.is/hsurj
"""

rsiadx = FunctionTransformer(RSI_ADX_optimizer, feature_names_out='one-to-one')

numeric_sub_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())])
categorical_sub_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
print(x_train.dtypes)
#hour/day are int32 under pandas 2.x on Windows, so select any float/integer dtype, not just float64/int64
#The PRICE-LEVEL features <OPEN>, <HIGH>, <LOW>, <CLOSE>, SMA and SAR are NOT model features (the talib step still
#uses the prices to compute RSI and ADX); VOL (500-bar rolling volatility) replaces them.
#Why: in 2000-2004 the price level mostly stood in for the volatility regime (corr(VOL, price) = -0.78: low USDCAD =
#volatile market = fewer "flat" bars). Raw levels drift without bound, so a model using them extrapolates outside its
#training price range; VOL carries the same information, and volatility regimes recur.
#Tested, original split (out-of-sample Sharpe, script units): level-free 0.78, with price levels 1.01, VOL 1.10.
#Walk-forward 2004-2021, 17 yearly blocks (annualized Sharpe): refitted yearly: level-free 1.03, levels 1.25, VOL 1.19;
#fitted once on 2000-2004: level-free 1.25, levels 1.34 (but it stopped predicting "flat" in 2007-2013), VOL 1.69.
#(None of these tests include transaction costs.)
PRICE_LEVELS = ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', 'SMA', 'SAR']
numeric_features_ix = x_train.select_dtypes(include=['floating']).columns.drop(PRICE_LEVELS)
categorical_features_ix = x_train.select_dtypes(include=['integer']).columns

numeric_features_no = []
for i in numeric_features_ix:
    numeric_features_no.append(x_train.columns.get_loc(i))
categorical_features_no = []
for i in categorical_features_ix:
    categorical_features_no.append(x_train.columns.get_loc(i))
col_no = []
for c in list(['<OPEN>','<HIGH>','<LOW>','<CLOSE>','RSI', 'ADX']):
    col_no.append(x_train.columns.get_loc(c))

#Note: transformer 3-element tuples can be: ('name', function or pipeline, column_number_list or column_index)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_sub_pipeline, list(numeric_features_ix)),
        ('cat', categorical_sub_pipeline, list(categorical_features_ix))], remainder='drop')
#remainder='drop' (was 'passthrough'): columns not selected above (the price levels) must be left out;
#with 'passthrough' they would silently be added back, unscaled, at the end
#Note: preprocessor selects columns by NAME because the talib step below reorders the columns
#(transformed columns first, then the remainder), which would make the positional numbers wrong.


"""
INSTRUCTIONS
Define a ColumnTransformer that uses the rsiadx tansformer
For guidance see: https://archive.is/hpwzH
"""


#pandas output + no name prefixes: RSI_ADX_optimizer receives a DataFrame and the output keeps the original column names
talib = ColumnTransformer(
    transformers=[('rsiadx', rsiadx, col_no)],
    remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')

"""
INSTRUCTIONS
select the correct SVC after checking if the target labels are balanced
"""

svc = LinearSVC(class_weight='balanced') #to balance the label categories, if they do not occur in equal numbers
svc = LinearSVC() 

"""
INSTRUCTIONS
Define a appropriate pipeline
For guidance see: https://archive.is/hpwzH
"""

pipe = Pipeline(steps=[('t', talib), ('preprocessor', preprocessor), ('svc', svc)])


c_rs = np.linspace(0.001, 1, num=8, endpoint=True) #1 default
c_rs = np.logspace(3, -4, num=50, endpoint = True)

"""
INSTRUCTIONS
Define a parameter grid for the timeperiod (try 5, 10 and 15)
for guidance see: https://archive.is/RPs90
"""

timeperiod_rs = [{'timeperiod': 5},{'timeperiod': 10},{'timeperiod': 15}]

param_grid = {'t__rsiadx__kw_args':  timeperiod_rs,'svc__C': c_rs}

grid_search = RandomizedSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True, random_state=rs)
#grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True)

grid_search.fit(x_train, y_train.values.ravel())

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters : {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score : {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("results_svc.csv")



#########################################################################################################################

# Train set
# Make "predictions" on training set (in-sample)
#positions = np.where(best_model.predict(x_train)> 0,1,-1 )
positions = np.where(grid_search.predict(x_train)> 0,1,-1 ) #POSITIONS


dailyRet = pd.Series(positions).fillna(0).values * df_train.retFut1 #for trading right after the open

dailyRet = dailyRet.fillna(0)

cumret = np.cumprod(dailyRet + 1) - 1


plt.figure(1)
plt.plot(cumret.index, cumret)
plt.title('Cross-validated SVC on currency: train set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(os.path.join('Results', '%s.png' %("TrainCumulative")))


cagr = (1 + cumret.iloc[-1]) ** (ANNUALIZATION / len(cumret)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret.values)
ratio = np.sqrt(ANNUALIZATION) * np.mean(dailyRet) / np.std(dailyRet)
print (('In-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD))

# Test set
# Make "predictions" on test set (out-of-sample)

#positions2 = np.where(best_model.predict(x_test)> 0,1,-1 )
positions2 = np.where(grid_search.predict(x_test)> 0,1,-1 ) #POSITIONS


dailyRet2 = pd.Series(positions2).fillna(0).values * df_test.retFut1 #for trading right after the open
dailyRet2 = dailyRet2.fillna(0)

cumret2 = np.cumprod(dailyRet2 + 1) - 1

plt.figure(2)
plt.plot(cumret2.index, cumret2)
title = 'Cross-validated SVC on currency: test set'
plt.title(title)
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(os.path.join('Results', '%s.png' %("TestCumulative")))

#metrics
acc_score = accuracy_score(y_test.values.ravel(), grid_search.predict(x_test))

#If this figure does not plot correctly select the lines and press F9 again
arr1 = y_test.values.ravel()
arr2 = grid_search.predict(x_test)
dfc = pd.DataFrame({'y_true': arr1, 'y_pred': arr2})
phi_k_corr = dfc.phik_matrix(interval_cols=[], njobs=1).iloc[1,0]
significance_overview = dfc.significance_matrix(interval_cols=[], njobs=1)
phi_k_sig  = dfc.significance_matrix(interval_cols=[], njobs=1).iloc[1,0]
phi_k_p_val = 1 - ndtr(phi_k_sig) 
plot_correlation_matrix(significance_overview.fillna(0).values, 
                        x_labels=significance_overview.columns, 
                        y_labels=significance_overview.index, 
                        vmin=-5, vmax=5, title="Significance of the coefficients", 
                        usetex=False, fontsize_factor=1.5, figsize=(7, 5))
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join('Results', '%s.png' %("PhikSignificance")))

cagr = (1 + cumret2.iloc[-1]) ** (ANNUALIZATION / len(cumret2)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret2.values)
ratio = np.sqrt(ANNUALIZATION) * np.mean(dailyRet2) / np.std(dailyRet2)
print (('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}  phi_k_corr={:0.6} phi_k_p_val={:0.6}  accuracy_score={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD, phi_k_corr, phi_k_p_val, acc_score))


#plot the residuals
true_y = y_test.values.ravel()
pred_y = grid_search.predict(x_test)
residuals = np.subtract(true_y, pred_y)

from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
fig, axes = plt.subplots(ncols=2, figsize=(14,4))
#sns.distplot is deprecated in seaborn 0.13: histogram (density) + fitted normal curve instead
sns.histplot(residuals, stat='density', ax=axes[0], label='Residuals')
xs = np.linspace(residuals.min(), residuals.max(), 200)
axes[0].plot(xs, norm.pdf(xs, *norm.fit(residuals)), color='k', label='Normal fit')
axes[0].set_xlabel('Residuals')
axes[0].set_title('Residual Distribution')
axes[0].legend()
plot_acf(residuals, lags=10, zero=False, ax=axes[1], title='Residual Autocorrelation')
axes[1].set_xlabel('Lags')
sns.despine()
fig.tight_layout();
#plt.show()
plt.savefig(os.path.join('Results', '%s.png' %("ResidualDistribution")))
plt.close("all")

#Residual autocorrelation
#If the p-value of the test is greater than the required significance (>0.05), residuals are independent
import statsmodels.api as sm
lb = sm.stats.acorr_ljungbox(residuals, lags=[10], boxpierce=False)
print("Ljung-Box test p-value", lb['lb_pvalue'].iloc[0]) #statsmodels >=0.13 returns a DataFrame

#white reality check
open_test = openp.iloc[10000:12000].reset_index(drop=True) #test rows; 0..n-1 index, aligned with pd.Series(positions2)
detrended_open = detrendPrice.detrendPrice(open_test)
#Row t: detrended price change D[t+1]-D[t] = (P[t+1]-P[t]) - trend slope, divided by the ACTUAL price P[t]
#(the price at the start of the return, same convention as retFut1 = P[t+1]/P[t]-1; equals retFut1[t] - slope/P[t]).
#Do NOT use detrended_open.pct_change(): it divides by the detrended level D[t], which detrendPrice shifts to a
#minimum just above 0 (~0.004-0.07 for USDCAD H3 blocks), inflating bar returns 16x-310x unevenly (and above +100%,
#which gives log(negative) = NaN in the bootstrap for short positions).
detrended_retFut1 = (detrended_open.diff().shift(-1) / open_test).fillna(0)
detrended_syst_rets = detrended_retFut1 * pd.Series(positions2).fillna(0)
np.random.seed(1) #reproducible bootstrap
WhiteRealityCheckFor1.bootstrap(detrended_syst_rets)
#plt.show() #an active plt.show() blocks a batch run until the plot window is closed
plt.savefig(os.path.join('Results', '%s.png' %("WhiteRealityCheck")))
plt.close("all")

#feature importance plot
#coef_ has one row per class and one column per transformed feature (incl. one-hot hour/day columns),
#so take the feature names from the fitted pipeline and plot the coefficients of the "up" class (label 1)
feature_names = best_model[:-1].get_feature_names_out()
up_row = list(best_model[-1].classes_).index(1)
importance = pd.DataFrame(zip(best_model[-1].coef_[up_row].tolist(), feature_names.tolist()))
importance.columns = ['slope','feature_name']
importance_plot = sns.barplot(x=importance['feature_name'], y=importance['slope'], data=importance,orient='v',dodge=False,order=importance.sort_values('slope',ascending=False).feature_name)
for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
    item.set_rotation(90)

#plt.show()
plt.savefig(os.path.join('Results', '%s.png' %("Coefficients")))

"""
This is how to obtain the continous predictors of a classifier
for use later on in Alphalens.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

#same talib + preprocessing steps (best RSI/ADX timeperiod, standard scaling, one-hot hour/day) and the tuned C
#as the best model, with a calibrated LinearSVC on top.
#(Previously: CalibratedClassifierCV(LinearSVC(C=0.001)) fitted on the raw x_train, i.e. with no scaling,
#no one-hot encoding, no RSI/ADX optimization and an untuned C, so it was a different model from the one backtested.)
model_svc = LinearSVC(C=best_parameters['svc__C'])
model = Pipeline(steps=[('t', clone(best_model[0])), ('preprocessor', clone(best_model[1])),
                        ('calibrated', CalibratedClassifierCV(model_svc))])

model.fit(x_train, y_train.values.ravel())
pred_class = model.predict(x_test)
pred_proba = model.predict_proba(x_test)
dfpred = pd.DataFrame(pred_proba, columns=["downs","statics","ups"])

dfpred["continuous_predictions"] = np.where(np.logical_and(dfpred["ups"]>dfpred["downs"],dfpred["ups"]>dfpred["statics"]), dfpred["ups"]+1, dfpred["statics"])
dfpred["continuous_predictions"] = np.where(np.logical_and(dfpred["downs"]>dfpred["ups"],dfpred["downs"]>dfpred["statics"]), dfpred["downs"]*-1, dfpred["continuous_predictions"])

"""
INSTRUCTION

using pip, install shap values

      - shap==0.51.0
      - slicer==0.0.8

change the LinearSVC in this script to a non-linear svc: NuSVC
svc = NuSVC(probability=False, cache_size=500) #rbf kernel by default; probability=True would enable predict_proba (slower)
use parameter search optimization to optimize NuSVC's nu parameter:
nu_rs = np.linspace(0.1, 0.9, num=5, endpoint=True)
Regarding the feature importance plot:
Comment out the feature importances plot.
Replace the commented-out-feature importances plot with one generated by Shap Values.
Make sure everthing runs.
Save the file as nu_scv_optim_talib.py 
Run linear_svc_optim_talib, linear_bayesian_optim_talib.py and nu_scv_optim_talib.py
After each run:
Transfer the files from Results to the appropriate Results directory.
Put the outputfile.txt and any .csv files in the appropriate Results directory.
Compare the results of linear_svc, nu_svc, and linear_bayesian especially 
the p value of White Reality Check in their corresponding ouputfile.txt
What causes this difference? (you can consult an AI)
Write a few lines about this and save it as comparison.txt, save it Results and turn it in with the rest.


"""

#####
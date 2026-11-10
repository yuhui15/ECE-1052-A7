
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

#df = pd.read_csv('EURUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('GBPUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('NZDUSD_H3_200001030000_202107201800.csv', sep='\t')
df = pd.read_csv('USDCAD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('USDCHF_H3_200001030000_202107201800.csv', sep='\t')

df['<DATETIME>'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df = df.set_index('<DATETIME>')
df.drop(['<TIME>'], axis=1, inplace=True)
df.drop(['<DATE>'], axis=1, inplace=True)

#save the open for white reality check
openp = df['<OPEN>'].copy()


#buld the best window features after the exploratory data analysis:
for n in list(range(1,15)):
    name = 'ret' + str(n)
    df[name] = df["<OPEN>"].pct_change(periods=n) #for trading with open
   

#build date-time features
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values

#build target assuming we know today's open
df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1).fillna(0) #if you enter the trade immediately after the open

#df = np.log(df+1)

"""
"""
#transform the target
#you have the option of a 3 label target or a 2 label target

#2 label target
#df['retFut1_categ']=0
#df.loc[df['retFut1']>df['retFut1'].iloc[:10000].quantile(q=0.51),'retFut1_categ']=1

#3 label target gets better results
df['retFut1_categ']=0
df.loc[df['retFut1']>df['retFut1'].iloc[:10000].quantile(q=0.66),'retFut1_categ']=1
df.loc[df['retFut1']<df['retFut1'].iloc[:10000].quantile(q=0.34),'retFut1_categ']=-1



#always check to see that the labels occur in equal numbers
dfgroups = df.groupby("retFut1_categ").count() #if they do not, use "balanced" parameter in the estimator(see below)

#Since we are trading right after the open, 
#we only know yesterday's  high low close volume spread etc.
df['<HIGH>'] = df['<HIGH>'].shift(1)
df['<LOW>'] = df['<LOW>'].shift(1)
df['<CLOSE>'] = df['<CLOSE>'].shift(1)
df['<VOL>'] = df['<VOL>'].shift(1)
df['<SPREAD>'] = df['<SPREAD>'].shift(1)

#Build some Ta-lib features, with n=10, to be optiized in the pipeline
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

#MixedNB requires the categorical features and the target to be encoded as 0,1,...,k-1
#One encoder per column, fitted on the train set only and reused on the test set, so a code means the same in both
#(re-fitting on the test set could silently give different codes if a category is missing from it).
#.copy() avoids pandas SettingWithCopy problems when assigning into slices of X and y.
from sklearn.preprocessing import LabelEncoder
x_train = x_train.copy()
x_test = x_test.copy()
y_train = y_train.copy()
y_test = y_test.copy()
le_hour = LabelEncoder().fit(x_train['hour'])
x_train['hour'] = le_hour.transform(x_train['hour'])
x_test['hour'] = le_hour.transform(x_test['hour'])
le_day = LabelEncoder().fit(x_train['day'])
x_train['day'] = le_day.transform(x_train['day'])
x_test['day'] = le_day.transform(x_test['day'])
le_y = LabelEncoder().fit(y_train['retFut1_categ']) #-1,0,1 -> 0,1,2
y_train['retFut1_categ'] = le_y.transform(y_train['retFut1_categ'])
y_test['retFut1_categ'] = le_y.transform(y_test['retFut1_categ'])

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
from sklearn.decomposition import PCA
import detrendPrice 
import WhiteRealityCheckFor1 
from mixed_naive_bayes import MixedNB #installed via the pip section of py3.11_AI_EXTENDED.yml (mixed-naive-bayes==0.0.3)
from sklearn.base import BaseEstimator, ClassifierMixin

#MixedNB is not a scikit-learn estimator (it does not inherit from BaseEstimator), so scikit-learn 1.6 cannot tell
#when a pipeline ending in MixedNB is fitted ("This Pipeline instance is not fitted yet" FutureWarning, an error
#from scikit-learn 1.8 on) and does not recognize it as a classifier. This thin wrapper fixes both; the model is unchanged.
class MixedNBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, categorical_features=None, max_categories=None, alpha=0.5, priors=None, var_smoothing=1e-9):
        self.categorical_features = categorical_features
        self.max_categories = max_categories
        self.alpha = alpha
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        y = np.asarray(y)
        self.model_ = MixedNB(categorical_features=self.categorical_features, max_categories=self.max_categories,
                              alpha=self.alpha, priors=self.priors, var_smoothing=self.var_smoothing)
        self.model_.fit(np.asarray(X), y)
        self.classes_ = np.unique(y) #MixedNB needs labels 0..k-1, so these are 0,1,...,k-1
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.classes_[self.model_.predict(np.asarray(X))]

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

#for optimizing the timeperiod of RSI and ADX
#x is a dataframe because this happens first in the pipeline
#(grid_search.fit receives the x_train DataFrame, so the ColumnTransformer passes a DataFrame with the
#original column names to this function)
def RSI_ADX_optimizer(x, timeperiod=None):
    x = x.copy() #do not modify the caller's (sliced) DataFrame in place
    if timeperiod is None: #ta-lib needs an int (None happens when kw_args is not set)
        timeperiod = 14 #ta-lib's default for RSI and ADX
    x['RSI'] = ta.RSI(x['<CLOSE>'].to_numpy(dtype=np.float64), timeperiod=timeperiod)
    x['ADX'] = ta.ADX(x['<HIGH>'].to_numpy(dtype=np.float64), x['<LOW>'].to_numpy(dtype=np.float64), x['<OPEN>'].to_numpy(dtype=np.float64), timeperiod=timeperiod)
    #return only RSI and ADX: the prices are used to compute them but are not model features (see below)
    return x[RSIADX_OUT].fillna(0) #the first rows of RSI/ADX are NaN, and these columns do not go through an imputer
RSIADX_OUT = ['RSI', 'ADX'] #the columns output by RSI_ADX_optimizer
#OLD numpy version (BUG: on a 2-D array x[3], x[4], x[5] select ROWS, not columns, so RSI/ADX were never
#recomputed, rows 4 and 5 of the data were overwritten, and the timeperiod search had no effect):
#def RSI_ADX_optimizer(x, timeperiod=None):
#    x[4] =ta.RSI(x[3], timeperiod=timeperiod) #x[0]x[1]x[2]x[3]x[4]x[5] are '<OPEN>','<HIGH>','<LOW>','<CLOSE>','RSI', 'ADX' 
#    x[5] = ta.ADX(x[1],x[2], x[0], timeperiod=timeperiod)
#    return np.nan_to_num(x)

#myscorer= "accuracy"  #same as None; 
#myscorer = make_scorer(phi_k, greater_is_better=True)
myscorer="f1_macro"
#my_scorer="f1_weighted" 
#myscorer = make_scorer(phi, greater_is_better=True) 

#when using smoother, use TimesSeriesSplit
#split = 5 
#split = TimeSeriesSplit(n_splits=5, max_train_size=2000) #fixed size window
split = TimeSeriesSplit(n_splits=5)

rsiadx = FunctionTransformer(RSI_ADX_optimizer)

numeric_sub_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())]) #see "Scaling" below: keeps MixedNB's var_smoothing from distorting the small-variance returns
categorical_sub_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))])

print(x_train.dtypes)
#select any float/integer dtype, not just float64/int64 (pandas 2.x may give e.g. int32 for hour/day)
#Excluded from the numeric features:
#- RSI and ADX: the rsiadx step below outputs them, recomputed with the timeperiod being optimized; passing the
#  precomputed (n=10) versions too would duplicate them, and naive Bayes would count their evidence twice
#  (it assumes the features are independent).
#The PRICE-LEVEL features <OPEN>, <HIGH>, <LOW>, <CLOSE>, SMA (moving average of the close) and SAR (a stop level in
#price units) are NOT model features; VOL (500-bar rolling volatility) replaces them, and the level-free RANGE, DSMA
#and DSAR carry their other useful information. Naive Bayes is especially hurt by raw levels: it scores each feature
#independently, so it cannot form price differences, and it counts the six near-identical price series as six
#independent pieces of evidence; at the test period's prices it never predicted "flat".
#Tested, original split (out-of-sample Sharpe, script units): with price levels 0.44 (CV f1_macro 28.8),
#level-free 0.67, level-free + VOL 0.84.
#Scaling: the numeric features (returns, Corr, OO, OC, RANGE, DSMA, DSAR, VOL) are standardized; the rsiadx outputs
#(RSI, ADX) are not. (The figures below were measured before RANGE, DSMA, DSAR and VOL were added.)
#In theory naive Bayes does not need scaling: it fits a separate Gaussian (mean, variance) per feature and class,
#and rescaling a feature multiplies every class's likelihood by the same factor, which cancels out.
#In practice the scaler still matters a little, because of MixedNB's var_smoothing: to avoid zero variances it adds
#var_smoothing (1e-9) x the LARGEST feature variance to every variance. That largest variance comes from the
#unscaled RSI (about 431 on the train set), so about 4.3e-7 is added to every variance. Without the scaler this
#would inflate the tiny variances of the unscaled features: ret14 +1.4% ... ret1 +17%, OO +8%, and
#OC +271% (OC = open minus previous close is usually tiny). Standardized (variance 1), the effect is negligible.
#(tested: without the scaler, CV f1_macro 40.17 -> 39.98 and 76 of the 2000 test predictions change). Keep the scaler.
#UPDATE: the PCA step added after the preprocessor (see pipe below) now standardizes all of these columns, RSI/ADX
#included, before rotating them; the scaler here is therefore redundant but harmless.
numeric_features_ix = x_train.select_dtypes(include=['floating']).columns.drop(['RSI', 'ADX', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', 'SMA', 'SAR'])
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
        ('rsiadx', rsiadx, col_no),    
        ('num', numeric_sub_pipeline, numeric_features_no),
        ('cat', categorical_sub_pipeline, categorical_features_no)], remainder='drop')
#remainder='drop' (was 'passthrough'): columns not selected by any transformer (SMA, SAR) must be left out;
#with 'passthrough' they would silently be added back, unscaled, at the end

#to inspect the preprocessed data:
#inspect_me = Pipeline(steps=[('preprocessor', preprocessor)]).set_params(preprocessor__rsiadx__kw_args={'timeperiod': 10}).fit_transform(x_train)
#the preprocessor outputs the rsiadx columns (RSI, ADX) first, then the numeric ones, then the categorical ones
#(hour, day), so the categorical columns come right after the first len(RSIADX_OUT)+len(numeric_features_no) columns
#(computed automatically below; [31,32] in the original version of this script)
first_cat = len(RSIADX_OUT) + len(numeric_features_no)
categorical_features_out = list(range(first_cat, first_cat + len(categorical_features_no)))
mnb = MixedNBClassifier(categorical_features=categorical_features_out) #previously hard-coded [31,32], found by inspection of inspect_me
#mnb.fit(w,y_train.values.ravel())

#PCA step (after the preprocessor): naive Bayes assumes the features are independent within each class, but the
#numeric inputs are strongly correlated (train set: ret1-OO 0.997, neighbouring returns ~0.94-0.96, ret6-DSMA 0.94,
#RSI ~ a sum of recent returns; mean within-class |corr| 0.41), so their evidence is counted several times.
#Scaler + PCA rotate ALL the numeric columns (RSI, ADX and the numeric features: the first first_cat columns of the
#preprocessor output) into uncorrelated components; all components are kept (a rotation, no reduction).
#RSI/ADX must be in the rotation too, which is why this is a separate step: they only exist after the rsiadx step.
#The scaler is needed so the unscaled RSI/ADX (variance ~431) don't dominate the PCA. No whitening: naive Bayes fits a
#variance per feature, so rescaling the components changes nothing (tested: identical results).
#hour/day pass through unchanged, at the same positions (categorical_features_out).
#Tested, original split: CV f1_macro 38.83 -> 39.98 (PCA on the numeric features without RSI/ADX: 39.08).
pca_step = ColumnTransformer(
    transformers=[
        ('pca', Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA())]), list(range(first_cat))),
        ('cat', 'passthrough', categorical_features_out)], remainder='drop')

pipe = Pipeline(steps=[('preprocessor', preprocessor),('pca', pca_step),('mnb', mnb)])

timeperiod_rs = [{'timeperiod': 5},{'timeperiod': 10},{'timeperiod': 15}]

param_grid = {'preprocessor__rsiadx__kw_args':  timeperiod_rs}

grid_search = RandomizedSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True, random_state = rs)
#grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True)

grid_search.fit(x_train, y_train.values.ravel())

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters : {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score : {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("results_mnb.csv")



#########################################################################################################################

# Train set
# Make "predictions" on training set (in-sample)
#positions = np.where(best_model.predict(x_train)> 0,1,-1 )
#MixedNB predicts the ENCODED labels 0,1,2, so decode them back to -1,0,1 before building positions
#(with encoded labels, "> 0" would also go long on the flat class: 0 -> code 1)
positions = np.where(le_y.inverse_transform(grid_search.predict(x_train))> 0,1,-1 ) #POSITIONS


dailyRet = pd.Series(positions).fillna(0).values * df_train.retFut1 #for trading right after the open

dailyRet = dailyRet.fillna(0)

cumret = np.cumprod(dailyRet + 1) - 1


plt.figure(1)
plt.plot(cumret.index, cumret)
plt.title('Results: train set')
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
positions2 = np.where(le_y.inverse_transform(grid_search.predict(x_test))> 0,1,-1 ) #POSITIONS


dailyRet2 = pd.Series(positions2).fillna(0).values * df_test.retFut1 #for trading right after the open
dailyRet2 = dailyRet2.fillna(0)

cumret2 = np.cumprod(dailyRet2 + 1) - 1

plt.figure(2)
plt.plot(cumret2.index, cumret2)
title = 'Results: test set'
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


# MixedNB does not have coefficients, instead it has a prior distribution
# -*- coding: utf-8 -*-
# -----------------------
# Importing (some) libraries
import pandas as pd
import numpy as np
import os
import time
from collections import defaultdict, Counter

from COVIRNA_ml_helpers_critical_vs_stable import *

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

no_iterations = 100 # for random splitting of data into train and test (default 20)
cv = 5 # cross-validation for hyper-parameter tuning (default 5)
FS_threshold = 90

# -------------------------------
# Prepare the results variables

results_avg = pd.DataFrame(columns=("AUC", "AUC_train", "balanced_accuracy", "Accuracy",
                                    "recall_critical", "recall_stable",
                                    "precision_critical", "precision_stable",
                                    "F1_critical", "F1_stable",))

results_std = pd.DataFrame()
results_CI = pd.DataFrame()
results_summary_AUC = pd.DataFrame()
results_summary_bal_acc = pd.DataFrame()

results_metric = {}
results_all = {}

features_all = {}
features_selected_all = Counter()
features_selected_all_afterCF = Counter()
features_selected_discovery_all = Counter()
features_selected_discovery_all_afterCF = Counter()

# -------------------------------
# Importing the data
dataset = pd.read_csv('./Dataset.csv')

# calculate correlation
from scipy.stats import pearsonr

corr_matrix_pearson = dataset.filter(regex='SEQ').corr(method="pearson")
corr_matrix_pearson.loc['SEQ0548','SEQ1056'] # 0.35076974873878203
corr_specific = pearsonr(dataset['SEQ0548'],dataset['SEQ1056'])
    
# Select only the necessary variables
X = dataset.filter(regex=r'(age|sex|bmi|smoker|SEQ)')
y = dataset[['outcome']]

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

le_sex = LabelEncoder()
le_sex.fit(X['sex'])
labels_sex = list(le_sex.classes_)
X.loc[:,'sex'] = le_sex.transform(X['sex'])

# manually encode smoker
labels_smoker = {'no':0, 'ex':1, 'yes':2}

X.loc[:,'smoker'] = X['smoker'].replace(labels_smoker)

le_outcome = LabelEncoder()
le_outcome.fit(y.values.ravel())
labels_outcome = list(le_outcome.classes_)
y = pd.DataFrame(le_outcome.transform(y.values.ravel()))
# list(le_outcome.inverse_transform(y))

### Stratified Shuffle sampling

indeces = []
indeces_undersampled = []

from sklearn.model_selection import StratifiedShuffleSplit
shuffle = StratifiedShuffleSplit(n_splits=no_iterations, test_size = 0.2)
for train_index, test_index in shuffle.split(X, y):
    indeces.append({})
    indeces[-1]["Train"] = train_index
    indeces[-1]["Test"] = test_index

# Time tracking
times_total_FS = []

print("########################################################")
print("# Starting with feature selection (step 1 out of 5)")
for iteration_FS in range(len(indeces)):
    time_start = time.time()
    iterations_left = len(indeces) - (iteration_FS+1)
    
    X_train, X_test, y_train, y_test, indeces_test = train_test_allocation(indeces, iteration_FS, X, y)
    
    ### undersample training data
    X_train, y_train, indeces_undersampled_all = undersample(X_train, y_train)
    
    # save the undersampled indeces to be used for ML
    indeces_undersampled.append({'Train':indeces_undersampled_all, 'Test': indeces_test})
    
    # scale the age and bmi data
    X_train, X_test = scale(X_train, X_test, ['age','bmi'])

    ### Boruta feature selection
    features_selected = list(boruta_FS(X_train, y_train))
    features_selected_all.update(features_selected)

    ### feature clustering
    clustered_features, _ = cluster_features(X_train[features_selected])
    features_selected_all_afterCF.update(clustered_features)

    # Time calculation
    times_total_FS = time_calculation(time_start, times_total_FS, iterations_left, iteration_FS)
   
# select the features above the percentage threshold (normalize for the number of iterations)
selected_features = [feature for feature, value in features_selected_all_afterCF.items() if (value/no_iterations*100) > FS_threshold]
# print(selected_features)

lncRNA_only = list(filter(lambda x: x != 'age', selected_features))

# Select only specific features
X_FS = X[selected_features]

# Time tracking
times_total_ML_imbalanced_FSbal = []

print("########################################################")
print("# Starting with ML on imbalanced data and selected FS features (step 2 out of 5)")
for iteration_ML in range(no_iterations):
    time_start = time.time()
    iterations_left = no_iterations - (iteration_ML+1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_FS, y, test_size=0.2, random_state=offset)
    
    # scale the age and bmi data (if selected for)
    X_train, X_test = scale(X_train, X_test, ['age','bmi'])
    
    # focus on all featuers
    results_metric_FS, results_all_FS = ML_pipeline(X_train, X_test, y_train, y_test, labels_outcome, subset='all-imbalanced-FSbal')

    fill_results_metric(results_metric_FS, results_metric)
    fill_results_all(results_all_FS, results_all)

    # focus only on lncRNA
    results_metric_age, results_all_age = ML_pipeline(X_train[lncRNA_only], X_test[lncRNA_only], y_train, y_test, labels_outcome, subset='lncRNA-imbalanced-FSbal')
    
    fill_results_metric(results_metric_age, results_metric)
    fill_results_all(results_all_age, results_all)
    
    # focus only on age
    results_metric_age, results_all_age = ML_pipeline(X_train[['age']], X_test[['age']], y_train, y_test, labels_outcome, subset='age-imbalanced-FSbal')
    
    fill_results_metric(results_metric_age, results_metric)
    fill_results_all(results_all_age, results_all)
    
    # Time calculation
    times_total_ML_imbalanced_FSbal = time_calculation(time_start, times_total_ML_imbalanced_FSbal, iterations_left, iteration_ML)

# Select only specific features
X_FS = X[selected_features]
feature_combos = generate_feature_combos(selected_features)

# Time tracking
times_total_ML_imbalanced_FSbal_combos = []

print("########################################################")
print("# Starting with ML on imbalanced data using feature combinations (step 3 out of 5)")
for iteration_ML in range(no_iterations):
    # iteration_ML = 0
    time_start = time.time()
    iterations_left = no_iterations - (iteration_ML+1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_FS, y, test_size=0.2)

    # scale the age and bmi data (if selected for)
    X_train, X_test = scale(X_train, X_test, ['age','bmi'])

    for subset, combo in feature_combos.items():
        
        results_metric_FS, results_all_FS = ML_pipeline(X_train[combo], X_test[combo], y_train, y_test, labels_outcome, subset=subset+'-imbalanced-FSbal', \
                                                        shap_imp=False, permutation_imp=False, save_model=False)
    
        fill_results_metric(results_metric_FS, results_metric)
        fill_results_all(results_all_FS, results_all)
    
    # Time calculation
    times_total_ML_imbalanced_FSbal_combos = time_calculation(time_start, times_total_ML_imbalanced_FSbal_combos, iterations_left, iteration_ML, divider=5)

# Select only specific features
X_FS = X[selected_features]

# Time tracking
times_total_ML = []

print("########################################################")
print("# Starting with machine learning on balanced data and selected FS features (step 4 out of 5)")
for iteration_ML in range(len(indeces_undersampled)):
    # iteration_ML = 0
    time_start = time.time()
    iterations_left = len(indeces_undersampled) - (iteration_ML+1)
    
    X_train, X_test, y_train, y_test, _ = train_test_allocation(indeces_undersampled, iteration_ML, X_FS, y)

    # scale the age and bmi data (if selected for)
    X_train, X_test = scale(X_train, X_test, ['age','bmi'])
    
    # focus on all featuers
    results_metric_FS, results_all_FS = ML_pipeline(X_train, X_test, y_train, y_test, labels_outcome, subset='all-balanced')

    fill_results_metric(results_metric_FS, results_metric)
    fill_results_all(results_all_FS, results_all)

    # focus only on lncRNA
    results_metric_age, results_all_age = ML_pipeline(X_train[lncRNA_only], X_test[lncRNA_only], y_train, y_test, labels_outcome, subset='lncRNA-balanced')
    
    fill_results_metric(results_metric_age, results_metric)
    fill_results_all(results_all_age, results_all)
    
    # focus only on age
    results_metric_age, results_all_age = ML_pipeline(X_train[['age']], X_test[['age']], y_train, y_test, labels_outcome, subset='age-balanced')
    
    fill_results_metric(results_metric_age, results_metric)
    fill_results_all(results_all_age, results_all)
    
    # Time calculation
    times_total_ML = time_calculation(time_start, times_total_ML, iterations_left, iteration_ML)
    
# Select only specific features
X_FS = X[selected_features]

# Time tracking
times_total_ML_combos = []

print("########################################################")
print("# Starting with ML on balanced data using feature combinations (step 5 out of 5)")
for iteration_ML in range(len(indeces_undersampled)):
    # iteration_ML = 0
    time_start = time.time()
    iterations_left = len(indeces_undersampled) - (iteration_ML+1)
    
    X_train, X_test, y_train, y_test, _ = train_test_allocation(indeces_undersampled, iteration_ML, X_FS, y)

    # scale the age and bmi data (if selected for)
    X_train, X_test = scale(X_train, X_test, ['age','bmi'])

    for subset, combo in feature_combos.items():
        
        results_metric_FS, results_all_FS = ML_pipeline(X_train[combo], X_test[combo], y_train, y_test, labels_outcome, subset=subset+'-balanced', \
                                                        shap_imp=False, permutation_imp=False, save_model=False)
    
        fill_results_metric(results_metric_FS, results_metric)
        fill_results_all(results_all_FS, results_all)
    
    # Time calculation
    times_total_ML_combos = time_calculation(time_start, times_total_ML_combos, iterations_left, iteration_ML, divider=5)    
    
import statsmodels.stats.api as sms

# summarize the results...
for algorithm in results_metric:
    for metric in results_metric[algorithm]:
        
        mean = round(np.mean(results_metric[algorithm][metric]),3)
        results_avg.loc[algorithm,metric] = mean
        
        std = round(np.std(results_metric[algorithm][metric]),3)
        combo_std = str(mean) + " ± " + str(std)
        results_std.loc[algorithm,metric] = combo_std
       
        CI_lower, CI_upper = sms.DescrStatsW(results_metric[algorithm][metric]).tconfint_mean()
        combo_CI = str(mean)+'('+str(round(CI_lower,3))+'-'+str(round(CI_upper,3))+')'
        results_CI.loc[algorithm,metric] = combo_CI
        
        if metric == 'AUC':
            info = algorithm.split('_')
            algorithm_summary = info[0]
            subset_summary = info[1]
            results_summary_AUC.loc[algorithm_summary,subset_summary] = mean
                
        if metric == 'balanced_accuracy':
            info = algorithm.split('_')
            algorithm_summary = info[0]
            subset_summary = info[1]
            results_summary_bal_acc.loc[algorithm_summary,subset_summary] = mean

# save the summary to excel
writer = pd.ExcelWriter('../Results.xlsx', engine='openpyxl')
results_summary_AUC.to_excel(writer, 'AUC_summary')
results_summary_bal_acc.to_excel(writer, 'bal_acc_summary')
results_avg.to_excel(writer, 'avg')
results_std.to_excel(writer, 'avg_std')
results_CI.to_excel(writer, 'avg_CI')
writer.close()

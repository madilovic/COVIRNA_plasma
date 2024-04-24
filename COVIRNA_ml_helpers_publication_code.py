# -*- coding: utf-8 -*-

import pandas as pd
from collections import defaultdict
import numpy as np
import time

from sklearn.metrics import classification_report, balanced_accuracy_score, make_scorer, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

import shap

def undersample(X, y):
    '''
    Undersample a dataset to the smallest class

    Parameters
    ----------
    X : pandas dataframe
    y : pandas dataframe containing only outcome column

    Returns
    -------
    X_undersampled
    y_undersampled
    indeces_undersampled_all : all indeces combined

    '''
    from itertools import chain
    
    # extract unique labels
    labels_outcome = y[0].unique()
    
    indeces = []
    indeces_undersampled = []

    # extract indeces
    for label in labels_outcome:
        indeces.append(y.index[y[0] == label].tolist())
                
    # determine the size of the smallest class
    smallest_class_size = min(map(len, indeces))
    
    # undersample all indeces to the size of the smallest class    
    for outcome_indeces in indeces:
        indeces_undersampled.append(list(np.random.choice(outcome_indeces, size=smallest_class_size, replace=False)))
        
    indeces_undersampled_all = list(chain.from_iterable(indeces_undersampled))
    
    X_undersampled = X.loc[indeces_undersampled_all]
    y_undersampled = y.loc[indeces_undersampled_all]
    
    return X_undersampled, y_undersampled, indeces_undersampled_all

def scale(X_train, X_test, column_names):
    '''
    Scale only the available columns
    
    '''
    from sklearn.preprocessing import StandardScaler
    
    for column_name in column_names:
        scaler = StandardScaler()
        try:
            column_train_scaled = scaler.fit_transform(X_train[[column_name]])
            column_test_scaled = scaler.transform(X_test[[column_name]])
            
            X_train.loc[:,column_name] = column_train_scaled
            X_test.loc[:,column_name] = column_test_scaled
        except:
            pass
    
    return X_train, X_test

def boruta_FS(X_train, y_train):
    '''
    Implement boruta for feature selection
    
    '''
    # warnings.filterwarnings("ignore")
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy

    # top_features_boruta = defaultdict(lambda: 0)
    
    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced')

    # fix the outdated numpy stuff
    np.int = np.int32
    np.float = np.float64
    np.bool = np.bool_
    
    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto')

    # find all relevant features
    feat_selector.fit(X_train.values, y_train.values.ravel())

    # check selected features
    features_selected = X_train.columns[feat_selector.support_]

    return(features_selected)

def random_forest_FS(X_train, y_train, pct_runs_threshold, no_iter = 100, print_run=False):
    '''
    Implement random forest (vanilla) for feature selection

    '''
    top10 = defaultdict(lambda: 0)

    for i in range(no_iter):
        if print_run==True:
            if i % 10 == 0:
                print('Random Forest Feature Selection Iteration:',i+1)

        # Fitting Random Forest to the training set
        from sklearn.ensemble import RandomForestClassifier
        
        classifier = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
        classifier.fit(X_train, y_train)
        
        importances = classifier.feature_importances_
        lncRNAs = list(X_train.columns)
        
        importances_df = pd.DataFrame(data = {'lncRNAs':lncRNAs,'importances':importances})
        importances_df.sort_values(by='importances', ascending=False, inplace=True)

        top10_lncRNAs = importances_df['lncRNAs'].values[:10]
        
        for lncRNA in top10_lncRNAs:
            top10[lncRNA] += 1
                
    top10_df = pd.DataFrame(top10.items(), columns=['lncRNA', 'no_runs'])
    # make percentage column
    top10_df['pct_runs'] = top10_df['no_runs'] / no_iter * 100
    lncRNAs_selected = top10_df[top10_df['pct_runs']>=pct_runs_threshold]['lncRNA'].values
    
    return(lncRNAs_selected)

def cluster_features(dataframe, corr_coef = 0.8, method = 'variance', corr_method = 'pearson'):
    '''
    Cluster features based on their correlation.
    
    Parameters
    ----------
    dataframe : pandas dataframe
        Input df containing feature values in columns.
        (Used for gene expression data: gene names in columns, patients in rows.)
    corr_coef : float
        Threshold for clustering of features into the same group. The default is 0.8.
    method : string
        Choose the method of selecting the best lncRNA from the cluster.
        'Variance' selects the feature with the highest variance.
        Additional statistical method, e.g. 'Wilcoxon' selects the feature with the highest Wilcoxon score compared to the dependent variable.
        The default is 'Variance'.
    corr_method: string
        The type of method used for correlation analysis.
        The default is 'pearson' (pandas also support ‘kendall’ or ‘spearman’).

    Returns
    -------
    List of features selected
    The dictionary contaning features and their corresponding cluster.

    '''
    feature_clusters = [] # list of clusters to be filled
    features_selected = [] # features to be returned
    features_checked = [] # keeps track of the features already checked

    ### first group features into clusters
    
    # perform correlation
    corr = dataframe.corr(method=corr_method)
    
    # iterate over every row of the dataset
    for current_feature, values in corr.iterrows():
        current_cluster = [] # keeps track of the current cluster
        # skip a feature if it has already been checked
        if current_feature in features_checked:
            continue
        # otherwise add it to the list of checked features
        else:
            features_checked.append(current_feature)
        # check only the features once and filter those with the selected coefficient
        values_current = values.drop(features_checked)
        values_current = values_current[values_current >= corr_coef]
        # if there are no features above the threshold correlation
        if len(values_current) == 0:
            # add a cluster with a single feature
            feature_clusters.append([current_feature])
        # else, make a cluster containing the current feature
        else:
            current_cluster.append(current_feature)
            # else go through the features above the threshold starting with the highest correlation
            values_current.sort_values(ascending=False, inplace=True)
            for feature in values_current.index:
                # add a feature to the cluster if it is correlated above the threshold
                # to all of the features currently in the cluster
                df_bool = dataframe[current_cluster+[feature]].corr() >= corr_coef
                if df_bool.all(axis=None):
                    current_cluster.append(feature)
                    # add the feature to the list of features checked
                    features_checked.append(feature)
            # add a cluster with selected features
            feature_clusters.append(current_cluster)
            
            # Note:
            # features not added to the cluster in this run will also not be added to the featuers checked
            # so their turn will come - to form their own new cluster ; or be added to the existing cluster    
    
    ### selected the desired features from each cluster
    if method == 'variance':
    
        features_variance = dataframe.var() # calculate variance of features
        
        # go over every cluster and select the feature with the highest variance
        for cluster in feature_clusters:
            if len(cluster) == 1:
                features_selected += cluster
            else:
                features_selected.append(features_variance[cluster].idxmax())
                               
    ### make a dictionary from the feature_clusters
    cluster_number = 1
    feature_clusters_dict = {}
    for cluster in feature_clusters:
        for feature in cluster:
            feature_clusters_dict[feature] = cluster_number
        cluster_number += 1
    
    return features_selected, feature_clusters_dict

def vif_FS(X_train, vif_threshold=10):
    '''
    Implement variance inflation factor for feature selection

    '''
    from statsmodels.stats.outliers_influence import variance_inflation_factor 
    
    vif_data = pd.DataFrame() 
    vif_data["feature"] = X_train.columns 
      
    # calculating VIF for each feature 
    vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) 
                              for i in range(len(X_train.columns))] 
      
    lncRNAs_selected = vif_data[vif_data['VIF']<=vif_threshold]['feature'].values
    
    return(lncRNAs_selected)

def generate_feature_combos(selected_features):
    '''
    Generate all feature combinations

    '''
    from itertools import combinations
    
    feature_combos = []
    feature_combos_dict = {}
    
    for r in range(1,len(selected_features)+1):
        feature_combos.extend(combinations(selected_features,r))
        
    for combination in feature_combos:
        combo = list(combination)
        subset = str(combination).replace("', '","-").strip("(',')")
        feature_combos_dict[subset] = combo
        
    return feature_combos_dict

def generate_forward_feature_combos(selected_features):
    '''
    Generate only feature combinations that incrementally update by one,
    based on the order of their appearence in selection method
    (i.e. the number of times they were selected using boruta)

    '''
    
    feature_combos_dict = {}
    
    for r in range(len(selected_features)):
        combo = selected_features[0:r+1]
        subset = "-".join(combo)
        
        feature_combos_dict[subset] = combo
        
    return feature_combos_dict
    

def train_test_allocation(indeces, iteration, X, y):
    indeces_train = indeces[iteration]["Train"]
    indeces_test = indeces[iteration]["Test"]
    X_train = X.iloc[indeces_train]
    y_train = y.iloc[indeces_train]
    X_test = X.iloc[indeces_test]
    y_test = y.iloc[indeces_test]
    
    return X_train, X_test, y_train, y_test, indeces_test

def fill_results_cr(cr, auc, auc_train, method, results_metric_subset):
    '''
    Parameters
    ----------
    cr : sklearn classification report
    auc : auc value for test
    auc_train : auc value for train
    method : type of method (e.g. predictors) used for ML
    results_metric_subset : variable that will contain results

    '''
    
    # extract specific values
    accuracy = cr['accuracy']
    recall_critical = cr['critical']['recall']
    recall_stable = cr['stable']['recall']
    balanced_accuracy = np.mean([recall_critical,recall_stable])
    precision_critical = cr['critical']['precision']
    precision_stable = cr['stable']['precision']
    F1_critical = cr['critical']['f1-score']
    F1_stable = cr['stable']['f1-score']
    
    # fill the results for the specific method
    if method not in results_metric_subset:
        results_metric_subset[method] = {"AUC":[], "AUC_train":[], "balanced_accuracy":[], "Accuracy":[],
                                "recall_critical":[], "recall_stable":[],
                                "precision_critical":[], "precision_stable":[],
                                "F1_critical":[], "F1_stable":[]}

    results_metric_subset[method]["AUC"].append(auc)
    results_metric_subset[method]["AUC_train"].append(auc_train)
    results_metric_subset[method]["balanced_accuracy"].append(balanced_accuracy)
    results_metric_subset[method]["Accuracy"].append(accuracy)
    results_metric_subset[method]["recall_critical"].append(recall_critical)
    results_metric_subset[method]["recall_stable"].append(recall_stable)
    results_metric_subset[method]["precision_critical"].append(precision_critical)
    results_metric_subset[method]["precision_stable"].append(precision_stable)
    results_metric_subset[method]["F1_critical"].append(F1_critical)
    results_metric_subset[method]["F1_stable"].append(F1_stable)
    
def fill_results_metric(results_metric_subset, results_metric):
    '''
    Fill results for every iteration

    '''
    for method, metric in results_metric_subset.items():
        if method not in results_metric:
            results_metric[method] = {"AUC":[], "AUC_train":[], "balanced_accuracy":[], "Accuracy":[],
                                    "recall_critical":[], "recall_stable":[],
                                    "precision_critical":[], "precision_stable":[],
                                    "F1_critical":[], "F1_stable":[]}
        for metric, iterations in metric.items():
            for iteration in iterations:
                results_metric[method][metric].append(iteration)
                
def fill_results_all_subset(results_all_subset, method, results_grouped):
    '''
    Fill in all results for every subset.

    '''
    dict_values = {'model':None, 'best_params':None, 'y_train':None, 'y_test':None, 'y_pred':None, 'y_proba':None, 'y_proba_train':None, 'cr':None, 'importance':None, 'imp_permutation':None}
    if method not in results_all_subset:
        results_all_subset[method] = dict_values
    for result, value in results_grouped.items():
        results_all_subset[method][result] = value

def fill_results_all(results_all_subset, results_all):
    '''
    Combine results from all subsets into a single variable.

    '''
    for method, results in results_all_subset.items():
        if method not in results_all:
            results_all[method] = {}
        for result, value in results.items():
            if result not in results_all[method]:
                results_all[method][result] = []
            results_all[method][result].append(value)

def fill_results_model(model, method, models_all_subset):
    models_all_subset[method].append(model)

def fill_results_importance(importance, method, importances_all_subset):
    importances_all_subset[method].append(importance)

def fill_results_importance_shap(importance_shap, method, importances_shap_all_subset):
    importances_shap_all_subset[method].append(importance_shap)

def fill_results_importance_permutation(importance_permutation, method, importances_permutation_all):
    importances_permutation_all[method].append(importance_permutation)
    
def fill_results_best_params(best_params, method, best_params_all_subset):
    best_params_all_subset[method].append(best_params)
    
def predict_test(model, X_train, X_test, y_train, y_test, labels):
    '''
    Calculate specific results.    

    '''

    y_pred = model.predict(X_test.values)
    try:
        y_proba = model.predict_proba(X_test)
        y_proba_train = model.predict_proba(X_train)
        AUC = roc_auc_score(y_test, y_proba[:, 1])
        AUC_train = roc_auc_score(y_train, y_proba_train[:, 1])
    except:
        y_proba = np.NAN
        y_proba_train = np.NAN
        AUC = np.NAN
        AUC_train = np.NAN
    
    cr = classification_report(y_test, y_pred, target_names = labels, output_dict=True)
    
    return y_pred, y_proba, y_proba_train, AUC, AUC_train, cr
    
    
def ML_pipeline(X_train, X_test, y_train, y_test, labels, subset='all', validate=False, \
                tuning=True, shap_imp=True, permutation_imp=False, cv=5, \
                    LR=True, KNN=True, SVM=True, GNB=True, XGB=True, MLP=True, save_model=True):
    '''
    Implement the whole ML pipeline here

    '''

    # results to be filled
    results_metric_subset = {}
    results_all_subset = {}
    models_all_subset = defaultdict(list)
    importances_all_subset = defaultdict(list)
    importances_shap_all_subset = defaultdict(list)
    importances_permutation_all = defaultdict(list)
    best_params_all_subset = defaultdict(list)

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    col_num = len(X_train.columns)
    
    y_train_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    
    ###########################################################################
    ############### Logistic Regression
    ###########################################################################
    results_grouped = {}
    
    # Fitting Logistic Regression to the training set
    from sklearn.linear_model import LogisticRegression
    classifier_logistic = LogisticRegression(max_iter = 10000, class_weight='balanced', n_jobs=-1)
    classifier_logistic.fit(X_train, y_train)
    
    model = classifier_logistic
        
    y_pred, y_proba, y_proba_train, AUC, AUC_train, cr = predict_test(model, X_train, X_test, y_train, y_test, labels)
    
    method = "Logistic_"+subset
    
    # calculate importance
    if col_num > 1:
        importance = classifier_logistic.coef_
        results_grouped['importance'] = importance
        if permutation_imp:
            importance_permutation = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=0)
            results_grouped['imp_permutation'] = importance_permutation
    
    if save_model:
        results_grouped.update({'model':model, 'best_params':None, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})
    else:
        results_grouped.update({'model':None, 'best_params':None, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})

    # adding results to variables
    fill_results_cr(cr, AUC, AUC_train, method, results_metric_subset)
    fill_results_all_subset(results_all_subset, method, results_grouped)
    
    ###########################################################################
    ############### Logistic Regression Parameter Tuning
    ###########################################################################
    results_grouped = {}
        
    # Fitting Logistic Regression to the training set
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    classifier_logistic_rand = LogisticRegression(max_iter = 10000, class_weight='balanced')
    grid_parameters = {'solver' : ['lbfgs','sag','saga'],
                         'C': [0.001,0.01,1],
                         'penalty': ['l1','l2','elasticnet']
                         }
    
    scoring = make_scorer(balanced_accuracy_score)
    classifier_logistic_grid = GridSearchCV(estimator = classifier_logistic_rand, param_grid = grid_parameters, scoring = scoring, cv=cv, verbose=0, n_jobs = -1)
    classifier_logistic_grid.fit(X_train, y_train)
    
    model = classifier_logistic_grid.best_estimator_
    best_params = classifier_logistic_grid.best_params_
            
    y_pred, y_proba, y_proba_train, AUC, AUC_train, cr = predict_test(model, X_train, X_test, y_train, y_test, labels)

    method = "Logistic_grid-"+subset
    
    # calculate importance
    if col_num > 1:
        importance = model.coef_
        results_grouped['importance'] = importance
        if permutation_imp:
            importance_permutation = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=0)
            results_grouped['imp_permutation'] = importance_permutation
            
    if save_model:
        results_grouped.update({'model':model, 'best_params':best_params, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})
    else:
        results_grouped.update({'model':None, 'best_params':best_params, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})

    # adding results to variables
    fill_results_cr(cr, AUC, AUC_train, method, results_metric_subset)
    fill_results_all_subset(results_all_subset, method, results_grouped)
            
    ###########################################################################
    ############### KNN #
    ###########################################################################
    results_grouped = {}

    # Fitting KNN to the training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier_knn = KNeighborsClassifier()
    classifier_knn.fit(X_train, y_train)
    
    model = classifier_knn
    
    y_pred, y_proba, y_proba_train, AUC, AUC_train, cr = predict_test(model, X_train.values, X_test, y_train, y_test, labels)

    method = "KNN_"+subset
       
    # calculate importance
    if col_num > 1:
        importance = False
        results_grouped['importance'] = importance
        if permutation_imp:
            importance_permutation = permutation_importance(model, X_train.values, y_train, n_repeats=5, random_state=0)
            results_grouped['imp_permutation'] = importance_permutation
            
    if save_model:
        results_grouped.update({'model':model, 'best_params':None, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})
    else:
        results_grouped.update({'model':None, 'best_params':None, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})

    # adding results to variables
    fill_results_cr(cr, AUC, AUC_train, method, results_metric_subset)
    fill_results_all_subset(results_all_subset, method, results_grouped)
    
    ###########################################################################
    ############### KNN Parameter Tuning #
    ###########################################################################
    results_grouped = {}

    # Fitting KNN to the training set
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    classifier_knn_rand = KNeighborsClassifier()
    grid_parameters = {'n_neighbors' : [5,10,15,20],
                         'leaf_size' : [10,30,50],
                         'p': [1,2]
                         }        
    
    scoring = make_scorer(balanced_accuracy_score)
    classifier_knn_grid = GridSearchCV(estimator = classifier_knn_rand, param_grid = grid_parameters, scoring = scoring, cv=cv, verbose=0, n_jobs = -1)
    classifier_knn_grid.fit(X_train, y_train)
    
    model = classifier_knn_grid.best_estimator_
    best_params = classifier_knn_grid.best_params_
    
    y_pred, y_proba, y_proba_train, AUC, AUC_train, cr = predict_test(model, X_train.values, X_test, y_train, y_test, labels)

    method = "KNN_grid-"+subset
                   
    # calculate importance
    if col_num > 1:
        importance = False
        results_grouped['importance'] = importance
        if permutation_imp:
            importance_permutation = permutation_importance(model, X_train.values, y_train, n_repeats=5, random_state=0)
            results_grouped['imp_permutation'] = importance_permutation
            
    if save_model:
        results_grouped.update({'model':model, 'best_params':best_params, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})
    else:
        results_grouped.update({'model':None, 'best_params':best_params, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})

    # adding results to variables
    fill_results_cr(cr, AUC, AUC_train, method, results_metric_subset)
    fill_results_all_subset(results_all_subset, method, results_grouped)
            
    ###########################################################################
    ############### SVM ###
    ###########################################################################
    results_grouped = {}

    # Fitting SVM to the training set
    from sklearn.svm import SVC
    classifier_svm = SVC(probability=True, class_weight='balanced')
    classifier_svm.fit(X_train, y_train)
    
    model = classifier_svm

    y_pred, y_proba, y_proba_train, AUC, AUC_train, cr = predict_test(model, X_train, X_test, y_train, y_test, labels)

    method = "SVM_"+subset
       
    # calculate importance
    if col_num > 1:
        importance = False
        # fill_results_importance_shap(importance_shap, method, importances_shap_all_subset)
        if permutation_imp:
            importance_permutation = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=0)
            results_grouped['imp_permutation'] = importance_permutation
                
    if save_model:
        results_grouped.update({'model':model, 'best_params':None, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})
    else:
        results_grouped.update({'model':None, 'best_params':None, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})

    # adding results to variables
    fill_results_cr(cr, AUC, AUC_train, method, results_metric_subset)
    fill_results_all_subset(results_all_subset, method, results_grouped)
    
    ###########################################################################
    ############### SVM Parameter Tuning ##
    ###########################################################################
    results_grouped = {}

    # Fitting SVM to the training set
    from sklearn.svm import SVC
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    classifier_svm_rand = SVC(probability=True, class_weight='balanced')
    grid_parameters = {'kernel' : ['poly','rbf'],
                         'C': [0.001,0.01,1],
                         'gamma': ['scale','auto'],
                         }
    
    scoring = make_scorer(balanced_accuracy_score)
    classifier_svm_grid = GridSearchCV(estimator = classifier_svm_rand, param_grid = grid_parameters, scoring = scoring, cv=cv, verbose=0, n_jobs = -1)
    classifier_svm_grid.fit(X_train, y_train)
    
    model = classifier_svm_grid.best_estimator_
    best_params = classifier_svm_grid.best_params_

    y_pred, y_proba, y_proba_train, AUC, AUC_train, cr = predict_test(model, X_train, X_test, y_train, y_test, labels)

    method = "SVM_grid-"+subset
  
    # calculate importance
    if col_num > 1:
        importance = False
        # fill_results_importance_shap(importance_shap, method, importances_shap_all_subset)
        if permutation_imp:
            importance_permutation = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=0)
            results_grouped['imp_permutation'] = importance_permutation

    if save_model:
        results_grouped.update({'model':model, 'best_params':best_params, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})
    else:
        results_grouped.update({'model':None, 'best_params':best_params, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})

    # adding results to variables
    fill_results_cr(cr, AUC, AUC_train, method, results_metric_subset)
    fill_results_all_subset(results_all_subset, method, results_grouped)
        
    ###########################################################################
    ############### Naive Bayes #
    ###########################################################################
    results_grouped = {}

    # Fitting Naive Bayes to the training set
    from sklearn.naive_bayes import GaussianNB
    classifier_gnb = GaussianNB()
    classifier_gnb.fit(X_train, y_train, sample_weight=y_train_weights)
    
    model = classifier_gnb

    y_pred, y_proba, y_proba_train, AUC, AUC_train, cr = predict_test(model, X_train, X_test, y_train, y_test, labels)

    method = "Naive_"+subset
  
    # calculate importance
    if col_num > 1:
        importance = False
        results_grouped['importance'] = importance
        if shap_imp:
            # explainer = shap.KernelExplainer(classifier_gnb.predict, shap.sample(X_train,100), check_additivity=False)
            explainer = shap.KernelExplainer(classifier_gnb.predict, X_train, check_additivity=False)
            # importance_shap = explainer(shap.sample(X_test,100))
            importance_shap = explainer(X_test)
            # importance_shap = importance_shap.values
            results_grouped['imp_shap'] = importance_shap
        if permutation_imp:
            importance_permutation = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=0)
            results_grouped['imp_permutation'] = importance_permutation
   
    if save_model:
        results_grouped.update({'model':model, 'best_params':None, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})
    else:
        results_grouped.update({'model':None, 'best_params':None, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})

    # adding results to variables
    fill_results_cr(cr, AUC, AUC_train, method, results_metric_subset)
    fill_results_all_subset(results_all_subset, method, results_grouped)
    
    ###########################################################################
    ############### Naive Bayes Parameter Tuning #
    ###########################################################################
    results_grouped = {}
    
    # Fitting Naive Bayes to the training set
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    classifier_gnb_rand = GaussianNB()
    grid_parameters = {'var_smoothing': [0.01,1e-5,1e-9]
                       }
    
    scoring = make_scorer(balanced_accuracy_score)
    classifier_gnb_grid = GridSearchCV(estimator = classifier_gnb_rand, param_grid = grid_parameters, scoring = scoring, cv=cv, verbose=0, n_jobs = -1)
    classifier_gnb_grid.fit(X_train, y_train, sample_weight=y_train_weights)
     
    model = classifier_gnb_grid.best_estimator_
    best_params = classifier_gnb_grid.best_params_
    
    y_pred, y_proba, y_proba_train, AUC, AUC_train, cr = predict_test(model, X_train, X_test, y_train, y_test, labels)
    
    method = "Naive_grid-"+subset
   
    # calculate importance
    if col_num > 1:
        importance = False
        results_grouped['importance'] = importance
        if shap_imp:
            # explainer = shap.KernelExplainer(classifier_gnb.predict, shap.sample(X_train,100), check_additivity=False)
            explainer = shap.KernelExplainer(classifier_gnb.predict, X_train, check_additivity=False)
            # importance_shap = explainer(shap.sample(X_test,100))
            importance_shap = explainer(X_test)
            # importance_shap = importance_shap.values
            results_grouped['imp_shap'] = importance_shap
        if permutation_imp:
            importance_permutation = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=0)
            results_grouped['imp_permutation'] = importance_permutation
   
    if save_model:
        results_grouped.update({'model':model, 'best_params':best_params, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})
    else:
        results_grouped.update({'model':None, 'best_params':best_params, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})
   
    # adding results to variables
    fill_results_cr(cr, AUC, AUC_train, method, results_metric_subset)
    fill_results_all_subset(results_all_subset, method, results_grouped)
        
  
    ###########################################################################
    ############### XGB #
    ###########################################################################
    results_grouped = {}

    # Fitting XGB to the training set
    from xgboost import XGBClassifier
    classifier_xgb = XGBClassifier()
    classifier_xgb.fit(X_train, y_train, sample_weight=y_train_weights)
    
    model = classifier_xgb

    y_pred, y_proba, y_proba_train, AUC, AUC_train, cr = predict_test(model, X_train, X_test, y_train, y_test, labels)

    method = "XGB_"+subset

    # calculate importance
    if col_num > 1:
        importance = classifier_xgb.feature_importances_
        results_grouped['importance'] = importance
        if permutation_imp:
            importance_permutation = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=0)
            results_grouped['imp_permutation'] = importance_permutation
 
    if save_model:
        results_grouped.update({'model':model, 'best_params':None, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})
    else:
        results_grouped.update({'model':None, 'best_params':None, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})

    # adding results to variables
    fill_results_cr(cr, AUC, AUC_train, method, results_metric_subset)
    fill_results_all_subset(results_all_subset, method, results_grouped)
    
    ###########################################################################
    ############### XGB Parameter Tuning #
    ###########################################################################
    results_grouped = {}

    # Fitting XGB to the training set
    from xgboost import XGBClassifier
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    classifier_xgb_rand = XGBClassifier()
    grid_parameters = {'n_estimators' : [50,100,200],
                         'max_depth' : [3,6,10],
                         'min_child_weight' : [1,2],
                         }
    
    scoring = make_scorer(balanced_accuracy_score)
    classifier_xgb_grid = GridSearchCV(estimator = classifier_xgb_rand, param_grid = grid_parameters, scoring = scoring, cv=cv, verbose=0, n_jobs = -1)
    
    classifier_xgb_grid.fit(X_train, y_train, sample_weight=y_train_weights)
    
    model = classifier_xgb_grid.best_estimator_
    best_params = classifier_xgb_grid.best_params_
    
    y_pred, y_proba, y_proba_train, AUC, AUC_train, cr = predict_test(model, X_train, X_test, y_train, y_test, labels)

    method = "XGB_grid-"+subset
 
    # calculate importance
    if col_num > 1:
        importance = model.feature_importances_
        results_grouped['importance'] = importance
        if permutation_imp:
            importance_permutation = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=0)
            results_grouped['imp_permutation'] = importance_permutation

    if save_model:
        results_grouped.update({'model':model, 'best_params':best_params, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})
    else:
        results_grouped.update({'model':None, 'best_params':best_params, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})

    # adding results to variables
    fill_results_cr(cr, AUC, AUC_train, method, results_metric_subset)
    fill_results_all_subset(results_all_subset, method, results_grouped)
            
 
    ###########################################################################
    ############### MLP #
    ###########################################################################
    results_grouped = {}

    # Fitting MLP to the training set
    from sklearn.neural_network import MLPClassifier
    classifier_mlp = MLPClassifier(max_iter=10000)
    classifier_mlp.fit(X_train, y_train)
    
    model = classifier_mlp

    y_pred, y_proba, y_proba_train, AUC, AUC_train, cr = predict_test(model, X_train, X_test, y_train, y_test, labels)

    method = "MLP_"+subset

    # calculate importance
    if col_num > 1:
        importance = False
        results_grouped['importance'] = importance
        if permutation_imp:
            importance_permutation = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=0)
            results_grouped['imp_permutation'] = importance_permutation

    if save_model:
        results_grouped.update({'model':model, 'best_params':None, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})
    else:
        results_grouped.update({'model':None, 'best_params':None, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})

    # adding results to variables
    fill_results_cr(cr, AUC, AUC_train, method, results_metric_subset)
    fill_results_all_subset(results_all_subset, method, results_grouped)
    
    ###########################################################################
    ############### MLP Parameter Tuning ##
    ###########################################################################
    results_grouped = {}

    # Fitting MLP to the training set
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    classifier_mlp_rand = MLPClassifier(max_iter=10000)
    grid_parameters = {'hidden_layer_sizes' : [(100,), (400,), (1000,), (400,400,400)],
                         'activation' : ['logistic', 'tanh', 'relu'],
                         'solver' : ['sgd', 'adam'],
                         'alpha': [1e-2,1e-4],
                         'learning_rate': ['constant','invscaling'],
                         }
    
    scoring = make_scorer(balanced_accuracy_score)
    classifier_mlp_grid = GridSearchCV(estimator = classifier_mlp_rand, param_grid = grid_parameters, scoring = scoring, cv=cv, verbose=0, n_jobs = -1)
    classifier_mlp_grid.fit(X_train, y_train)
    
    model = classifier_mlp_grid.best_estimator_
    best_params = classifier_mlp_grid.best_params_

    y_pred, y_proba, y_proba_train, AUC, AUC_train, cr = predict_test(model, X_train, X_test, y_train, y_test, labels)

    method = "MLP_grid-"+subset
 
    # calculate importance
    if col_num > 1:
        importance = False
        results_grouped['importance'] = importance
        if permutation_imp:
            importance_permutation = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=0)
            results_grouped['imp_permutation'] = importance_permutation
   
    if save_model:
        results_grouped.update({'model':model, 'best_params':best_params, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})
    else:
        results_grouped.update({'model':None, 'best_params':best_params, 'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba, 'y_proba_train':y_proba_train, 'cr':cr})

    # adding results to variables
    fill_results_cr(cr, AUC, AUC_train, method, results_metric_subset)
    fill_results_all_subset(results_all_subset, method, results_grouped)
        
    ##################
    # Return results #
    ##################
    return results_metric_subset, results_all_subset

def time_calculation(time_start, times_total, iterations_left, iteration, divider=10):
    time_end = time.time()
    time_total = round((time_end-time_start)/60,2)
    times_total.append(time_total)
    # print("### Iteration runtime:",time_total,"minutes")
        
    time_left = (np.average(times_total)/60*iterations_left)
    if (iteration+1) % divider == 0:
        if time_left < 1:
            print("# Iteration", iteration+1, "completed. Estimated minutes left:", round(time_left*60,2))
        else:
            print("# Iteration", iteration+1, "completed. Estimated hours left:", round(time_left,2))
        
        # warnings.filterwarnings("ignore")
    return times_total
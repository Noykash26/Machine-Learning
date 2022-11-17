## import ##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.common import random_state
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import statsmodels.formula.api as smf
from numpy import mean
from numpy import std
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.svm import  SVC,LinearSVC
from sklearn.metrics import roc_auc_score, davies_bouldin_score, confusion_matrix, silhouette_score
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.ensemble import RandomForestClassifier
import datetime
from sklearn.metrics import precision_score , roc_curve

# K-Medoids
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm
import matplotlib.pyplot as plt

                                                                                                            
#####################################################################################################################
' read data from part A '
#####################################################################################################################

df = pd.read_csv('C:/Users/User/Desktop/ML_project/new_data.csv', header=0)
df.shape
df = df.drop(df.columns[0], axis=1) # remove first column

#####################################################################################################################
'  Model Training '
#####################################################################################################################

## shuffle the samples ##
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

## K-fold ##
k_folds = KFold(n_splits=10, random_state=(42), shuffle=True)

## split data ##

X_train = df.drop('target', 1)
y_train = df['target']

#####################################################################################################################
'  1. Decision tree '
#####################################################################################################################

#####################################################################################################################
'  1.1 Default Decision tree '
#####################################################################################################################

model_dt = DecisionTreeClassifier(criterion='entropy', random_state=42)                                
train_result_dt = pd.DataFrame()
valid_result_dt = pd.DataFrame()

for train_index , val_index in k_folds.split(X_train): 
    model_dt.fit(X_train.iloc[train_index] , y_train.iloc[train_index])
    # score of both train and validation sets
    roc_auc_train = roc_auc_score(y_train.iloc[train_index], model_dt.predict(X_train.iloc[train_index])) 
    roc_auc_val = roc_auc_score(y_train.iloc[val_index],  model_dt.predict(X_train.iloc[val_index])) 
    train_result_dt = train_result_dt.append({'Train ROC-AUC':roc_auc_train},ignore_index=True)
    valid_result_dt = valid_result_dt.append({'Validation ROC-AUC':roc_auc_val},ignore_index=True)
    
avg_roc_auc_train = train_result_dt.mean()
avg_roc_auc_val = valid_result_dt.mean()
print(avg_roc_auc_train)
print(avg_roc_auc_val)

#####################################################################################################################
'  1.2. Hyperparameter tuning '
#####################################################################################################################
                                                                                     
param_grid = {'criterion': ['entropy', 'gini'],
              'splitter' : ['best','random'],
              'max_depth': np.arange(1, 35, 1),
              'min_samples_split': np.arange(10, 500, 10),
              #'ccp_alpha': np.arange(0, 1, 0.05) - no alpha is needed
             }

## number of combinations of parameters
comb = 1
for list_ in param_grid.values():
    comb *= len(list_)
print("number of combinations is: " ,comb)

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, refit=True, cv=k_folds, return_train_score=True, scoring='roc_auc')
grid_search.fit(X_train,y_train)
Results = pd.DataFrame(grid_search.cv_results_)
print('The best parameters are:' ,grid_search.best_params_)
results_grid_search1 = pd.DataFrame(Results).sort_values('rank_test_score')[['params', 'mean_test_score']]
results_grid_search2 = pd.DataFrame(Results).sort_values('mean_train_score',ascending=False)[['params', 'mean_train_score']]


## Max depth influence on accuracy - graph
max_depth_range= np.arange(1, 50 , 1)
result_maxDepth = pd.DataFrame()
train_result_maxDepth = pd.DataFrame()
valid_result_maxDepth = pd.DataFrame()
for max_depth in max_depth_range:
    train_result_maxDepth = pd.DataFrame(None)
    valid_result_maxDepth = pd.DataFrame(None)
    for train_index, val_index in k_folds.split(X_train):
        model_maxDepth=DecisionTreeClassifier(criterion='gini', max_depth=max_depth, splitter='best', ccp_alpha = 0, min_samples_split=370, random_state=42) # our best parameters after tuning!
        model_maxDepth.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        acc_train_maxDepth = roc_auc_score(y_train.iloc[train_index], model_maxDepth.predict(X_train.iloc[train_index]))  # roc auc of the train
        acc_val_maxDepth = roc_auc_score(y_train.iloc[val_index], model_maxDepth.predict(X_train.iloc[val_index]))  # roc auc of the validation
        train_result_maxDepth = train_result_maxDepth.append({'Train accuracy': acc_train_maxDepth}, ignore_index=True)
        valid_result_maxDepth = valid_result_maxDepth.append({'Validation accuracy': acc_val_maxDepth}, ignore_index=True)
    avg_train_maxDepth = train_result_maxDepth.mean()
    avg_val_maxDepth = valid_result_maxDepth.mean()
    result_maxDepth = result_maxDepth.append({'max_depth': max_depth,
                                'train_acc':avg_train_maxDepth,
                                 'val_acc':avg_val_maxDepth}, ignore_index=True)

plt.figure(figsize=(13, 4))
plt.plot(result_maxDepth['max_depth'], result_maxDepth['train_acc'], marker='o', markersize=4)
plt.plot(result_maxDepth['max_depth'], result_maxDepth['val_acc'], marker='o', markersize=4)
plt.title("Infulence of Max depth on accuracy")
plt.ylabel('Accuracy', fontsize = 10)
plt.xlabel('Max depth', fontsize = 10)
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.show()
                                                                                                                      
#####################################################################################################################
'  1.3. Best Decision Tree After Tuning '


#best_model = grid_search.best_estimator_
#preds = best_model.predict(X_test)
#print("Test accuracy: ", round(accuracy_score(y_test, preds), 3))

#####################################################################################################################

best_dt = DecisionTreeClassifier(criterion='gini', max_depth=6, splitter='best', ccp_alpha = 0.0, min_samples_split=370, random_state=42)                
train_result_best = pd.DataFrame()
valid_result_best = pd.DataFrame()
for train_index, val_index in k_folds.split(X_train):
    best_dt.fit(X_train.iloc[train_index], y_train.iloc[train_index])                                                    
    roc_auc_train = roc_auc_score(y_train.iloc[train_index] , best_dt.predict(X_train.iloc[train_index]))
    roc_auc_val = roc_auc_score(y_train.iloc[val_index],  best_dt.predict(X_train.iloc[val_index]))
    train_result_best = train_result_best.append({'Train Best ROC-AUC':roc_auc_train},ignore_index=True)
    valid_result_best = valid_result_best.append({'Validation Best ROC-AUC':roc_auc_val},ignore_index=True)
avg_roc_auc_train_best = train_result_best.mean()
avg_roc_auc_val_best = valid_result_best.mean()
print(avg_roc_auc_train_best)
print(avg_roc_auc_val_best)

## Tree printing
plt.figure(figsize=(14.2, 7))
plot_tree(best_dt, filled=True, max_depth=3, class_names=['0', '1'], feature_names=X_train.columns, fontsize=9)
plt.show()

## Feature importances
feature_importance = best_dt.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0])
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx], fontsize=10)
plt.title('Feature Importance', fontsize = 20)
plt.show()

#####################################################################################################################
'  2. ANN '
#####################################################################################################################

## Standartization                                                                               # TODO - what about categorical standartization?
X_train_S = X_train.copy() 
col_names = ['experience', 'city_development_index']
features = X_train_S[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
X_train_S[col_names] = features

#####################################################################################################################
'  2.1. Default ANN '
#####################################################################################################################

ann_model = MLPClassifier(random_state=42)                                                    # TODO - activate = logistic?

ann_train_result = pd.DataFrame()
ann_valid_result = pd.DataFrame()

for train_idx, val_idx in k_folds.split(X_train_S): 
    x_trainSet = X_train_S.iloc[train_idx]
    y_trainSet = y_train.iloc[train_idx]                                                      # TODO - why iloc only for y?
    x_valSet = X_train_S.iloc[val_idx]
    y_valSet = y_train.iloc[val_idx]

    ann_model.fit(x_trainSet, y_trainSet)
    roc_auc_train = roc_auc_score(y_trainSet, ann_model.predict(x_trainSet))
    roc_auc_val = roc_auc_score(y_valSet, ann_model.predict(x_valSet))
    ann_train_result = ann_train_result.append({'Train Best ROC-AUC': roc_auc_train}, ignore_index=True)
    ann_valid_result = ann_valid_result.append({'Validation Best ROC-AUC': roc_auc_val}, ignore_index=True)

ann_avg_train = ann_train_result.mean()
ann_avg_valid = ann_valid_result.mean()
print(ann_avg_train)
print(ann_avg_valid)
plt.plot(ann_model.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

plt.plot(ann_valid_result)
plt.plot(ann_train_result)
plt.title('Validation vs Train - ROC-AUC', fontsize=20)                                                        # validation vs train - accuracy graph                      
plt.show()

#####################################################################################################################
'  2.2. Hyperparameter tuning '
#####################################################################################################################

param_grid = {'hidden_layer_sizes': [(8,), (8,8),(8,8,8),(8,16),(8,32),(16,),(16,16),(16,16,16),(16,32),(32,),(32,32),(32,32,32) ],
              'max_iter': [500],
              'activation': ['logistic'],
              'learning_rate_init': np.arange(0.001, 0.05, 0.01),
              'learning_rate': ['constant', 'adaptive']                                                         # TODO - add alpha if overfitting
              }                                                                                                 # TODO - max iter? graphs of parameters?
                                                                                                                # TODO - where are kfolds?
grid_search = GridSearchCV(estimator=MLPClassifier(random_state=42, verbose=False, max_iter=500), param_grid=param_grid,
                           refit=True, cv=k_folds, return_train_score=True, scoring='roc_auc')                  # TODO - scoring in this place is correct?
grid_search.fit(X_train_S, y_train)
Results = pd.DataFrame(grid_search.cv_results_)
print('The best parameters are:', grid_search.best_params_)
y_val = Results['mean_test_score']
y_trainAnn = Results['mean_train_score']
results_grid_search1 = pd.DataFrame(Results).sort_values('rank_test_score')[['params', 'mean_test_score']]
results_grid_search2 = pd.DataFrame(Results).sort_values('mean_train_score', ascending=False)[['params', 'mean_train_score']]

plt.plot(y_val)
plt.plot(y_trainAnn)
plt.title('Validation vs Train - ROC-AUC', fontsize=20)                                                        # validation vs train - accuracy graph                      
plt.show()

## confusion matrix
print(confusion_matrix(y_true=y_train, y_pred=grid_search.predict(X_train_S)))                                    # TODO - only on train?

#####################################################################################################################
'  3. SVM '
#####################################################################################################################

## Standartization for continouts features (categorical features handled in part A)
X_train_svm = X_train.copy() 
col_names = ['experience', 'city_development_index']
features = X_train_svm[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
X_train_svm[col_names] = features

#####################################################################################################################
'  3.1. Default SVM '
#####################################################################################################################

model_svm = SVC(kernel='linear',decision_function_shape='ovr',random_state=42)             # ovr - for binary classification
train_result_svm = pd.DataFrame()
valid_result_svm = pd.DataFrame()
predictions = pd.DataFrame()
for train_index, val_index in k_folds.split(X_train_svm):                                 
    model_svm.fit(X_train_svm.iloc[train_index], y_train.iloc[train_index])                    
    roc_auc_train_svm = roc_auc_score(y_train.iloc[train_index], model_svm.predict(X_train_svm.iloc[train_index]))
    roc_auc_val_svm = roc_auc_score(y_train.iloc[val_index], model_svm.predict(X_train_svm.iloc[val_index]))
    train_result_svm = train_result_svm.append({'Train ROC-AUC':roc_auc_train_svm}, ignore_index=True)
    valid_result_svm = valid_result_svm.append({'Validation ROC-AUC':roc_auc_val_svm}, ignore_index=True)
    
avg_roc_auc_train_svm = train_result_svm.mean()
avg_roc_auc_val_svm = valid_result_svm.mean()
print(avg_roc_auc_train_svm)
print(avg_roc_auc_val_svm)
print('The coef is:')
print(model_svm.coef_)
print('The intercept is:')
print(model_svm.intercept_)

#####################################################################################################################
'  3.2. Hyperparameter tuning '
#####################################################################################################################

param_grid = {'C': [0.01,0.1,0.5,1,3,10,100],
              'decision_function_shape': ['ovr', 'ovo'],
              'gamma': ['scale', 'auto']                              
             }                                                                                                
grid_search = GridSearchCV(estimator=SVC(kernel='linear', decision_function_shape="ovr", random_state=42), param_grid=param_grid, refit=True, cv=k_folds , return_train_score=True, scoring='roc_auc')
grid_search.fit(X_train_svm,y_train)
Results = pd.DataFrame(grid_search.cv_results_)
print('The best parameters are:' ,grid_search.best_params_)
y_val = Results['mean_test_score']
y_trainSVC = Results['mean_train_score']
results_grid_search1 = pd.DataFrame(Results).sort_values('rank_test_score')[['params', 'mean_test_score']]
results_grid_search2 = pd.DataFrame(Results).sort_values('mean_train_score',ascending=False)[['params', 'mean_train_score']]

#####################################################################################################################
'  3.3. Best linear SVM After Tuning  '
#####################################################################################################################

model_bestsvm = SVC(kernel='linear', C=100, gamma = 'scale', random_state=42)
train_result_bestsvm = pd.DataFrame()
valid_result_bestsvm= pd.DataFrame()
for train_index , val_index in k_folds.split(X_train_svm):
    model_bestsvm.fit(X_train_svm[train_index] , y_train.iloc[train_index])                                            # TODO - why iloc only or y      
    roc_auc_train_bestsvm = roc_auc_score(y_train.iloc[train_index],  model_bestsvm.predict(X_train_svm[train_index])) #accuracy of the train
    roc_auc_val_bestsvm = roc_auc_score(y_train.iloc[val_index],  model_bestsvm.predict(X_train_svm[val_index])) #accuracy of the validation
    train_result_bestsvm = train_result_bestsvm.append({'Train Best ROC-AUC':roc_auc_train_bestsvm},ignore_index=True)
    valid_result_bestsvm = valid_result_bestsvm.append({'Validation Best ROC-AUC':roc_auc_val_bestsvm},ignore_index=True)

print('The Weights of the Features are:')
print("[CDI , relevant_experience , enrolled_university , experience , company_size , company_type_0 , company_type_1 , last_new_job, is_working, university_relevant_exp_0, university_relevant_exp_1")

# printing the 'Betaot'                                                   
print(model_bestsvm.coef_)
print('The intercept is:')
print(model_bestsvm.intercept_)

avg_roc_auc_train_bestsvm = train_result_bestsvm.mean()
avg_roc_auc_val_bestsvm = valid_result_bestsvm.mean()
print(avg_roc_auc_train_bestsvm)
print(avg_roc_auc_val_bestsvm)

#####################################################################################################################
'  4. K - Medoids '
#####################################################################################################################

#####################################################################################################################
'  4.1 Prepare the data '
#####################################################################################################################

## Standartization for continouts features (categorical features handled in part A)                                                             
X_train_km_df = X_train.copy() 
col_names = ['experience', 'city_development_index']
features = X_train_km_df[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
X_train_km_df[col_names] = features
X_train_km_df.drop('Y', axis='columns', inplace=True)
X_train_km = np.array(X_train_km_df)

metric = distance_metric(type_metric.GOWER, max_range=X_train_km.max(axis=0))
dbi_list = []
sil_list = []
max_n_clusters = 10

#####################################################################################################################
'  4.2. Tuning K '
#####################################################################################################################

for n_clusters in tqdm(range(2, max_n_clusters, 1)):
    initial_medoids = kmeans_plusplus_initializer(X_train_km, max_n_clusters).initialize(return_index=True)
    kmedoids_instance = kmedoids(X_train_km, initial_medoids, metric=metric)
    kmedoids_instance.process()
    assignment = kmedoids_instance.predict(X_train_km)

    sil = silhouette_score(X_train_km, assignment)
    dbi = davies_bouldin_score(X_train_km, assignment)

    dbi_list.append(dbi)
    sil_list.append(sil)
    
#####################################################################################################################
'  4.3. Graphs - Silhouette, Davies-bouldin'
#####################################################################################################################

plt.plot(range(2, max_n_clusters, 1), sil_list, marker='o')
plt.title("Silhouette")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(2, max_n_clusters, 1), dbi_list, marker='o')
plt.title("Davies-bouldin")
plt.xlabel("Number of clusters")
plt.show()

j=2
for dbs in dbi_list:
    print('For k=',j, ', Davies-Bouldin score is:', dbs)
    j+=1

j=2
for sil in sil_list:
    print('For k=',j, ', Silhouette score is:', sil)
    j+=1

#####################################################################################################################
'  4.4. clustering with the best k , k=4 '
#####################################################################################################################

initial_medoids = kmeans_plusplus_initializer(X_train_km, 4).initialize(return_index=True)
kmedoids_instance = kmedoids(X_train_km, initial_medoids, metric=metric)
kmedoids_instance.process()
assignment = kmedoids_instance.predict(X_train_km)
results = pd.concat([X_train_km_df, pd.DataFrame(assignment, columns = ['Cluster'])], axis= 1)
print(results)
results.to_csv("clusters.csv")

#####################################################################################################################
'  Evaluation '
#####################################################################################################################

## evaluate SVM                                  
best_svm = SVC(kernel='linear', decision_function_shape="ovr", C = 100, gamma = 'scale', random_state=42)
roc_auc_svm = cross_val_score(best_svm, X_train_svm, y_train, cv=k_folds, scoring='roc_auc')
print(roc_auc_svm.mean())
# F1_svm = cross_val_score(best_svm, X_train_svm, y_train, cv=k_folds, scoring='f1')
# print (F1_svm.mean()) ----> not working, we will calculate 
recall = cross_val_score(best_svm, X_train_svm, y_train, cv=k_folds, scoring='recall_macro')
print (recall.mean())
precision = cross_val_score(best_svm, X_train_svm, y_train, cv=k_folds, scoring='precision_macro')
print (precision.mean())
# calculating F1
sum = 0
for i in range (1,10):
    sum += (2*precision[i]*recall[i])/(precision[i]+recall[i])
mean_f = sum/10
print(mean_f)

## evaluate DT
best_dt = DecisionTreeClassifier(max_depth=7, ccp_alpha=0, criterion='gini', splitter='best', min_samples_split=370, random_state=42)
roc_auc_dt = cross_val_score(best_dt, X_train, y_train, cv=k_folds, scoring='roc_auc')
print(roc_auc_dt.mean())
F1_dt = cross_val_score(best_dt, X_train, y_train, cv=k_folds, scoring='f1')
print (F1_dt.mean())

## evaluate ANN                                    
best_ann = MLPClassifier(activation='logistic', hidden_layer_sizes= (16,32), max_iter= 500, learning_rate_init=0.011, random_state=42)
roc_auc_ann = cross_val_score(best_ann, X_train_S, y_train, cv=k_folds, scoring='roc_auc')
print("roc auc mean", roc_auc_ann.mean())
F1_ann = cross_val_score(best_ann, X_train_S, y_train, cv=k_folds, scoring='f1')
print ("f1 mean", F1_ann.mean())

#####################################################################################################################
'  Improvments - 1. random forest with Hyperparameters Tuning '
#####################################################################################################################

from sklearn.ensemble import RandomForestClassifier

## max depth
param_grid = {'max_depth': np.arange(1, 30, 1)}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, refit=True, cv=k_folds , return_train_score=True, scoring='roc_auc')
grid_search.fit(X_train,y_train)
Results = pd.DataFrame(grid_search.cv_results_)
print('The best parameters are:', grid_search.best_params_)
results_grid_search1 = pd.DataFrame(Results).sort_values('rank_test_score')[['params', 'mean_test_score']]
results_grid_search2 = pd.DataFrame(Results).sort_values('mean_train_score',ascending=False)[['params', 'mean_train_score']]

## criteria + alpha
param_grid = {'criterion': ['entropy', 'gini'],
              'ccp_alpha': np.arange(0, 1, 0.05)}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42, max_depth=8), param_grid=param_grid, refit=True, cv=k_folds , return_train_score=True, scoring='roc_auc')
grid_search.fit(X_train,y_train)
Results = pd.DataFrame(grid_search.cv_results_)
print('The best parameters are:' ,grid_search.best_params_)
results_grid_search1 = pd.DataFrame(Results).sort_values('rank_test_score')[['params', 'mean_test_score']]
results_grid_search2 = pd.DataFrame(Results).sort_values('mean_train_score',ascending=False)[['params', 'mean_train_score']]

## number of trees in the forest
param_grid = {'n_estimators': np.arange(10, 150, 2)}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42, max_depth=8, criterion='entorpy', ccp_alpha=0), param_grid=param_grid, refit=True, cv=k_folds , return_train_score=True, scoring='roc_auc')
grid_search.fit(X_train,y_train)
Results = pd.DataFrame(grid_search.cv_results_)
print('The best parameters are:' ,grid_search.best_params_)
results_grid_search1 = pd.DataFrame(Results).sort_values('rank_test_score')[['params', 'mean_test_score']]
results_grid_search2 = pd.DataFrame(Results).sort_values('mean_train_score',ascending=False)[['params', 'mean_train_score']]

## running the best random forest tree
randomForest = RandomForestClassifier(random_state=42, max_depth=8, criterion='entropy', ccp_alpha=0, n_estimators = 88) 

for train_index , val_index in k_folds.split(X_train):
    train_result_rf = pd.DataFrame(None)
    valid_result_rf = pd.DataFrame(None)
    randomForest.fit(X_train.iloc[train_index] , y_train.iloc[train_index])
    roc_auc_train = roc_auc_score(y_train.iloc[train_index], randomForest.predict(X_train.iloc[train_index])) # roc auc of the train
    roc_auc_val = roc_auc_score(y_train.iloc[val_index],  randomForest.predict(X_train.iloc[val_index])) # roc auc of the validation
    train_result_rf = train_result_rf.append({'Train accuracy':roc_auc_train}, ignore_index=True)
    valid_result_rf = valid_result_rf.append({'Validation accuracy':roc_auc_val}, ignore_index=True)
avg_roc_auc_train = train_result_rf.mean()
avg_roc_auc_val = valid_result_rf.mean()
print(avg_roc_auc_train)
print(avg_roc_auc_val)

#####################################################################################################################
'  Improvments - 2. Smote '
#####################################################################################################################
# sudo pip install imbalanced-learn

# decision tree  on imbalanced dataset with SMOTE oversampling and random undersampling 
# + grid search k value for SMOTE oversampling for imbalanced classification

from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


# define dataset
X_train_smote = df.drop('target', 1)
y_train_smote = df['target']
print(Counter(y_train_smote)) # 0.33

# values to evaluate
k_values = [1, 2, 3, 4, 5, 6, 7]
for k in k_values:

	# define pipeline
	model = RandomForestClassifier()
	over = SMOTE(sampling_strategy=0.4, k_neighbors=5)
	under = RandomUnderSampler(sampling_strategy=0.5)
	steps = [('over', over), ('under', under), ('model', model)]
	pipeline = Pipeline(steps=steps)
    
	# evaluate pipeline
	# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(pipeline, X_train_smote, y_train_smote, scoring='roc_auc', cv=k_folds)
	score = mean(scores)
	print('> k=%d, Mean ROC AUC: %.3f' % (k, score))

# model with best K
model = RandomForestClassifier(random_state=42)
over = SMOTE(sampling_strategy=0.4, k_neighbors=5)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under), ('model', model)]
pipeline = Pipeline(steps=steps)

## Hyperparameter
# max depth - 8
param_grid = {'model__max_depth': np.arange(1, 30, 1)}
grid_search = GridSearchCV(pipeline, param_grid=param_grid, refit=True, cv=k_folds, return_train_score=True, scoring='roc_auc')
grid_search.fit(X_train,y_train)
Results = pd.DataFrame(grid_search.cv_results_)
print('The best parameters are:', grid_search.best_params_)
results_grid_search1 = pd.DataFrame(Results).sort_values('rank_test_score')[['params', 'mean_test_score']]
results_grid_search2 = pd.DataFrame(Results).sort_values('mean_train_score',ascending=False)[['params', 'mean_train_score']]

model = RandomForestClassifier(random_state=42, max_depth=8)
over = SMOTE(sampling_strategy=0.4, k_neighbors=5)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under), ('model', model)]
pipeline = Pipeline(steps=steps)

## criteria + alpha - gini, 0
param_grid = {'model__criterion': ['entropy', 'gini'],
              'model__ccp_alpha': np.arange(0, 1, 0.05)}
grid_search = GridSearchCV(pipeline, param_grid=param_grid, refit=True, cv=k_folds , return_train_score=True, scoring='roc_auc')
grid_search.fit(X_train,y_train)
Results = pd.DataFrame(grid_search.cv_results_)
print('The best parameters are:' ,grid_search.best_params_)
results_grid_search1 = pd.DataFrame(Results).sort_values('rank_test_score')[['params', 'mean_test_score']]
results_grid_search2 = pd.DataFrame(Results).sort_values('mean_train_score',ascending=False)[['params', 'mean_train_score']]

model = RandomForestClassifier(random_state=42, max_depth=8, criterion='gini', ccp_alpha=0.0)
over = SMOTE(sampling_strategy=0.4, k_neighbors=5)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under), ('model', model)]
pipeline = Pipeline(steps=steps)

## number of trees in the forest - 140
param_grid = {'model__n_estimators': np.arange(10, 150, 2)}
grid_search = GridSearchCV(pipeline, param_grid=param_grid, refit=True, cv=k_folds , return_train_score=True, scoring='roc_auc')
grid_search.fit(X_train,y_train)
Results = pd.DataFrame(grid_search.cv_results_)
print('The best parameters are:' ,grid_search.best_params_)
results_grid_search1 = pd.DataFrame(Results).sort_values('rank_test_score')[['params', 'mean_test_score']]
results_grid_search2 = pd.DataFrame(Results).sort_values('mean_train_score',ascending=False)[['params', 'mean_train_score']]

#grid_search.best_estimator_.predict(X_test)[:,1]
#####################################################################################################################
'  Final predictions '
#####################################################################################################################

X_test = pd.read_csv('C:/Users/User/Desktop/ML_project/X_test_preprocessing.csv', header=0)
X_test = X_test.drop(X_test.columns[0], axis=1) 

## running the random forset on the test dataset
randomForest = RandomForestClassifier(random_state=42, max_depth=8, criterion= 'entropy', ccp_alpha=0, n_estimators = 60) # our best parameters
randomForest.fit(X_train, y_train)
y_predictions = randomForest.predict_proba(X_test)[:,1] # probability of being class=1
pd.DataFrame(y_predictions, columns=['y probs']).to_csv('G3_ytest.csv')

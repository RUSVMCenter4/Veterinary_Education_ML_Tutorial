"""
This is the Python code found within the manuscript 
" "
as well as the additional code needed to create the random forest machine learning models with the datasets missing GRE values.
@author: Sarah Hooper
"""

#Import dataset.
dataset = pd.read_excel(r'C:\location of the downloaded dataset\MockData.xlsx')

#Import required packages and/or required functions:
import random 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from imblearn.over_sampling import SMOTE 
from collections import Counter  
from sklearn.metrics import plot_roc_curve
import shap

    
#Data Preprocessing
#Create student IDs
randomList = []

for i in range(5000):
        d = random.randint(1,400)
        
        if d not in randomList:
                randomList.append(d)

len(randomList)

#Append this to the current pandas dataframe dataset
dataset.insert(0, "RandomID", randomList)
#Drop out names
dataset.drop(['Full Name'], axis=1, inplace=True) #axis 1 means column, 0 would mean rows.  inplace means to drop operation in same dataframe vs creating a copy after drop

#View dataset
dataset

#Prepare data for analysis

#We don't need the student ID for this ML exercise so we can drop it from the dataframe
dataset.drop(['RandomID'], axis=1, inplace=True)


#To one-hot encode for race column
dataset_OneHot = pd.get_dummies(dataset, columns=["Race"], drop_first=False)
dataset_OneHot.head()

#To dummy encode for Gender column
dataset_OneHot = pd.get_dummies(dataset_OneHot, columns=["Gender"], drop_first=True)
print(dataset_OneHot.head())

#Allow us to see all the columns in the dataset
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)

#Now review our initial first few rows of the dataset
dataset_OneHot.head()


#We need to now break our dataset into the variables and the target column
#Note order of the columns from the above review of the dataset.

X = dataset_OneHot.loc[:, dataset_OneHot.columns != 'Fail']  #create dataframe without target, [rows, columns]
y = dataset_OneHot['Fail'] #target variable for prediction

#Oversampling to allow 0 and 1 target to be equal
X_resampled, y_resampled = SMOTE(random_state=23).fit_resample(X, y)
print("length of X_resampled", len(X_resampled))
print("length of y_resampled", len(y_resampled))
print(sorted(Counter(y_resampled).items()))  #Shows count for 0 vs 1 target
X_trainSMOTE, X_testSMOTE, y_trainSMOTE, y_testSMOTE = train_test_split(X_resampled, y_resampled, stratify=y_resampled, test_size=0.3, random_state=50) #70% of data training, 30% of data testing

#Check sizes of arrays to make sure it they match each other
print('Training Variables Shape:', X_trainSMOTE.shape)
print('Training Target Shape:', y_trainSMOTE.shape)
print('Testing Variables Shape:', X_testSMOTE.shape)
print('Testing Target Shape:', y_testSMOTE.shape)

#Build base model without any changes to default settings
forest_base = RandomForestClassifier(random_state=(23))
#Fit data to model via.fit
forest_base.fit(X_trainSMOTE, y_trainSMOTE) #using training data
y_predictions = forest_base.predict(X_testSMOTE) #Make predictions using testing data set
y_trueSMOTE = y_testSMOTE #True values of test dataset

kf = KFold(shuffle=True, n_splits=5)
cv_scores = cross_val_score(forest_base, X_resampled, y_resampled, cv=kf,error_score='raise')
cv_scores
cv_scores.mean()

#Mean values for each parameter

score_accuracy_mean = cross_val_score(forest_base, X_testSMOTE, y_trueSMOTE, cv=kf, scoring='accuracy').mean()
print(score_accuracy_mean)

score_auc_mean = cross_val_score(forest_base, X_testSMOTE, y_trueSMOTE, cv=kf, scoring = 'roc_auc').mean()
print(score_auc_mean)

score_precision_mean = cross_val_score(forest_base, X_testSMOTE, y_trueSMOTE, cv=kf, scoring='precision').mean()
print(score_precision_mean)

score_recall_mean = cross_val_score(forest_base, X_testSMOTE, y_trueSMOTE, cv=kf, scoring = 'recall').mean()
print(score_recall_mean)

score_f1_mean = cross_val_score(forest_base, X_testSMOTE, y_trueSMOTE, cv=kf, scoring='f1').mean()
print(score_f1_mean)

scoring = make_scorer(recall_score, pos_label=0)
score_specificity_mean = cross_val_score(forest_base, X_testSMOTE, y_trueSMOTE, cv=kf, scoring = scoring).mean()
cross_val_score(forest_base, X_testSMOTE, y_trueSMOTE, cv=kf, scoring = scoring)
print(score_specificity_mean)


#Most important features from best model
feature_imp_base = pd.Series(forest_base.feature_importances_, index=X.columns)
feature_imp_base = feature_imp_base.sort_values(ascending=False)
feature_imp_base


##Assess hyperparamters to try to improve upon base model:
#First create our hyperparameter grid
# Number of trees to be included in random forest
n_trees = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 50)]  #will create 50 trees b/w 100 and 2000

# Number of features to consider at every split
max_features = ['sqrt','None']  #auto will consider the max features = sqrt(n_features) wheras none will consider all features

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 200, num = 20)]
max_depth.append(None) #Also adding none which means that the nodes are expanded until all leaves are pure or until all leaves contain less than the min_samples_split samples.

# Minimum number of samples required to split a node
min_samples_split = [2, 4, 6, 8, 10]  #default is 2

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 6, 8, 10] #default is 1

# Method of selecting samples for training each tree to include using bootstrap method AND to also then try without bootsrap meaning the whole dataset is used to build each tree.
bootstrap = [True, False]

# Create the random grid
hyper_grid = {'n_estimators': n_trees,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(hyper_grid)

#Initiate base model to tune
best_params = RandomForestClassifier(random_state=(23))

#Use random grid search to find best hyperparameters, uses kfold validation as cross validation method
#Search 200 different combinations

best_params_results = RandomizedSearchCV(estimator=best_params, param_distributions= hyper_grid, n_iter=200,
                                         cv=kf, verbose=5, random_state=(23))

#Fit the random search model
best_params_results.fit(X_trainSMOTE, y_trainSMOTE)

#Find the best parameters from the grid search results
best_params_results.best_params_

#Results
#{'n_estimators': 332,
# 'min_samples_split': 2,
# 'min_samples_leaf': 1,
# 'max_features': 'sqrt',
# 'max_depth': 180,
# 'bootstrap': True}

#Build another hyperparameter grid using narrowed down parameter guidelines from above
#Then use GridSearchCV method to search every combinatino of grid
new_grid = {'n_estimators': [250, 275, 300, 325, 332, 350, 375],
               'max_features': ['sqrt'],
               'max_depth': [160, 165, 170, 175, 180, 185, 190, 195],
               'min_samples_split': [1, 2, 3, 4, 5, 6],
               'min_samples_leaf': [1, 2, 3],
               'bootstrap': [True]}
print(new_grid)

best_params = RandomForestClassifier(random_state=(23))
best_params_grid_search = GridSearchCV(estimator=best_params, param_grid=new_grid, cv=kf, n_jobs=-1, verbose=10)
best_params_grid_search.fit(X_trainSMOTE, y_trainSMOTE)

best_params_grid_search.best_params_

###Results to use for new model
#{'bootstrap': True,
# 'max_depth': 160,
# 'max_features': 'sqrt',
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'n_estimators': 375}

#Using the results of the best parameters, we will create a new model and show the specific arguments.

best_grid_model = RandomForestClassifier(n_estimators=375, max_features='sqrt', max_depth=(160), min_samples_split=2, min_samples_leaf=2, bootstrap=True)

#Best model based upon grid
best_grid_model.fit(X_trainSMOTE, y_trainSMOTE)

y_predictions = best_grid_model.predict(X_testSMOTE) #Make predictions using testing data set
y_true = y_testSMOTE #True values of test dataset

#Mean values for each parameter

acc = cross_val_score(best_grid_model, X_testSMOTE, y_testSMOTE, cv=kf, scoring='accuracy').mean()
print(acc)

auc = cross_val_score(best_grid_model, X_testSMOTE, y_testSMOTE, cv=kf, scoring = 'roc_auc').mean()
print(auc)

precision = cross_val_score(best_grid_model, X_testSMOTE, y_testSMOTE, cv=kf, scoring='precision').mean()
print(precision)

recall = cross_val_score(best_grid_model, X_testSMOTE, y_testSMOTE, cv=kf, scoring = 'recall').mean()
print(recall)

f1 = cross_val_score(best_grid_model, X_testSMOTE, y_testSMOTE, cv=kf, scoring='f1').mean()
print(f1)

scoring = make_scorer(recall_score, pos_label=0)
specificity = cross_val_score(best_grid_model, X_testSMOTE, y_testSMOTE, cv=kf, scoring = scoring).mean()
print(specificity)

#To plot ROC curve
plot_roc_curve(best_grid_model, X_testSMOTE, y_testSMOTE)  

###best model

#Most important features from best model
feature_imp = pd.Series(best_grid_model.feature_importances_, index=X.columns)
feature_imp = feature_imp.sort_values(ascending=False)
feature_imp

#Plot Most important features over 1%
feature_imp.nlargest(4).plot(kind='bar', title="Most Important Features for RUSVM Students Admitted in ", style='seaborn-colorblind')




####
#To look at how biased missing low GRE scores impact the outcome

#Choose one line of imported data and remove pound sign immediately before biased_dataset.  After creation of base random forest model, then replace pound sign 
#and remove the second pound sign to use the same code to run the second dataset.

#biased_dataset = pd.read_excel(r'C:\location of saved dataset\MockData.xlsx', sheet_name='BiasedGRE1')
#biased_dataset = pd.read_excel(r'C:\location of saved dataset\MockData.xlsx', sheet_name='BiasedGRE2')

#Drop out names
biased_dataset.drop(['Full Name'], axis=1, inplace=True) #axis 1 means column, 0 would mean rows.  inplace means to drop operation in same dataframe vs creating a copy after drop

#View dataset
biased_dataset

#Prepare data for analysis

#To one-hot encode for race column
biased_dataset_OneHot = pd.get_dummies(biased_dataset, columns=["Race"], drop_first=False)
biased_dataset_OneHot.head()

#To dummy encode for Gender column
biased_dataset_OneHot = pd.get_dummies(biased_dataset_OneHot, columns=["Gender"], drop_first=True)
print(biased_dataset_OneHot.head())

#We need to deal with our missing values.  We could eliminate the lines, replace a
#biased_dataset_OneHot.dropna(axis=0, how='any', inplace=True)
biased_dataset_OneHot.fillna((biased_dataset_OneHot['GRE'].mean()), inplace=True)

#We need to now break our dataset into the variables and the target column
#Note order of the columns from the above review of the dataset.

X_biased = biased_dataset_OneHot.loc[:, biased_dataset_OneHot.columns != 'Fail']  #create dataframe without target, [rows, columns]
y_biased = biased_dataset_OneHot['Fail'] #target variable for prediction

#Oversampling to allow 0 and 1 target to be equal
X_resampledB, y_resampledB = SMOTE(random_state=23).fit_resample(X_biased, y_biased)
print("length of X_resampledB", len(X_resampledB))
print("length of y_resampledB", len(y_resampledB))
print(sorted(Counter(y_resampledB).items()))  #Shows count for 0 vs 1 target
X_trainSMOTE_Biased, X_testSMOTE_Biased, y_trainSMOTE_Biased, y_testSMOTE_Biased = train_test_split(X_resampledB, y_resampledB, stratify=y_resampledB, test_size=0.3, random_state=50) #70% of data training, 30% of data testing

#Check sizes of arrays to make sure it they match each other
print('Training Variables Shape:', X_trainSMOTE_Biased.shape)
print('Training Target Shape:', y_trainSMOTE_Biased.shape)
print('Testing Variables Shape:', X_testSMOTE_Biased.shape)
print('Testing Target Shape:', y_testSMOTE_Biased.shape)

#Build base model without any changes to default settings
forest_base_Biased = RandomForestClassifier(random_state=(23))
#Fit data to model via.fit
forest_base_Biased.fit(X_trainSMOTE_Biased, y_trainSMOTE_Biased) #using training data
y_predictions = forest_base_Biased.predict(X_testSMOTE_Biased) #Make predictions using testing data set
y_trueSMOTE = y_testSMOTE_Biased #True values of test dataset

kf = KFold(shuffle=True, n_splits=5)
cv_scores = cross_val_score(forest_base_Biased, X_resampledB, y_resampledB, cv=kf,error_score='raise')
cv_scores
cv_scores.mean()

#Mean values for each parameter

score_accuracy_mean = cross_val_score(forest_base_Biased, X_testSMOTE_Biased, y_trueSMOTE, cv=kf, scoring='accuracy').mean()
print(score_accuracy_mean)

score_auc_mean = cross_val_score(forest_base_Biased, X_testSMOTE_Biased, y_trueSMOTE, cv=kf, scoring = 'roc_auc').mean()
print(score_auc_mean)

score_precision_mean = cross_val_score(forest_base_Biased, X_testSMOTE_Biased, y_trueSMOTE, cv=kf, scoring='precision').mean()
print(score_precision_mean)

score_recall_mean = cross_val_score(forest_base_Biased, X_testSMOTE_Biased, y_trueSMOTE, cv=kf, scoring = 'recall').mean()
print(score_recall_mean)

score_f1_mean = cross_val_score(forest_base_Biased, X_testSMOTE_Biased, y_trueSMOTE, cv=kf, scoring='f1').mean()
print(score_f1_mean)

scoring = make_scorer(recall_score, pos_label=0)
score_specificity_mean = cross_val_score(forest_base_Biased, X_testSMOTE_Biased, y_trueSMOTE, cv=kf, scoring = scoring).mean()
cross_val_score(forest_base_Biased, X_testSMOTE_Biased, y_trueSMOTE, cv=kf, scoring = scoring)
print(score_specificity_mean)


#Most important features from best model
feature_imp_base_biased = pd.Series(forest_base_Biased.feature_importances_, index=X.columns)
feature_imp_base_biased = feature_imp_base_biased.sort_values(ascending=False)
feature_imp_base_biased


###
#How feature of importance is calculated by SHAP values
#Using already defined best model with all GRE data from above.
best_grid_model 

#Most important features from best model
shap_feature_imp = shap.TreeExplainer(best_grid_model)
shap_values = shap_feature_imp.shap_values(X_testSMOTE)
shap.summary_plot(shap_values, X_testSMOTE)


shap.summary_plot(shap_values[1], X_testSMOTE)  #Fail
shap.summary_plot(shap_values[0], X_testSMOTE)  #No Fail

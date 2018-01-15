import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from imblearn.over_sampling import SMOTE

def get_opt_model(clf, param_grid, X_train, y_train, scoring = 'accuracy'):
    '''
    PARAMETERS:
    clf(classifier): classifier to run GridSearch on
    param_grid(dict): dictionary of parameters to test in GridSearch
    X_train(array): array of predictor variables
    y_train(array): array of response variable
    scoring: score to maximize during GridSearch, default to accuracy

    PRINTS: best score and best parameter values determined by GridSearch
    RETURNS: fitted model (based on entire train set) with best parameter values
    '''

    best_clf = GridSearchCV(clf, param_grid, scoring = 'accuracy', cv = 5, n_jobs = -1)
    best_clf.fit(X_train, y_train)
    print(best_clf.best_score_)
    print(best_clf.best_params_)
    return best_clf

def print_confusion_matrix(best_clf, X_test, y_test):
    '''
    PARAMETERS:
    best_clf(classifier): classifier with optimal parameters determined by GridSearch
    X_test(array): array of test set predictor variables
    y_test(array): array of test set response variable

    PRINTS: confusion matrix
    '''

    #predictions made by model
    clf_predicts = best_clf.predict(X_test)
    print(confusion_matrix(y_test, clf_predicts))

if __name__ == '__main__':

    #data separated with ;
    winedf = pd.read_csv('../data/winequality-red.csv', sep = ';')

    #numpy array of quality rating response variable
    y = winedf.pop('quality').values

    #remove free sulfur dioxide column since it provides
    #redundant information with total sulfur dioxide column
    winedf.pop('free sulfur dioxide')

    X = winedf.values
    #create train-test splits then scale the predictor variables
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    '''
    testing random forest classifier
    '''
    # rf = RandomForestClassifier()
    # rf_test_params = {'n_estimators': [25, 30, 35, 40], 'max_depth': [10, 15, 20, 25], \
    #             'min_samples_leaf': [1, 5]}
    # best_rf = get_opt_model(rf, rf_test_params, X_train_sc, y_train)
    # print_confusion_matrix(best_rf, X_test_sc, y_test)

    # 0.694745621351
    #
    # {'n_estimators': 35, 'max_depth': 15, 'min_samples_leaf': 1}
    #
    #    3 [[  0   0   1   0   0   0]
    # T  4  [  0   0   8   5   0   0]
    # R  5  [  0   0 122  41   1   0]
    # U  6  [  0   0  34 120  14   1]
    # E  7  [  0   0   0  18  29   1]
    #    8  [  0   0   0   2   3   0]]
    #          3   4   5   6   7   8
    #               PREDICTED
    #

    '''
    testing gradient boosting classifier
    '''
    gb = GradientBoostingClassifier()
    gb_test_params = {'n_estimators': [100, 500, 750], 'learning_rate': [.01, .05, .1, .2], \
                    'min_samples_leaf': [1, 5, 7]}
    best_gb = get_opt_model(gb, gb_test_params, X_train_sc, y_train)
    print_confusion_matrix(best_gb, X_test_sc, y_test)

    # 0.656380316931
    #
    # {'n_estimators': 500, 'learning_rate': 0.05, 'min_samples_leaf': 5}
    #
    #    3 [[ 0   0   1   0   0   0]
    # T  4 [  0   1   8   4   0   0]
    # R  5 [  1   0 123  39   1   0]
    # U  6 [  0   2  33 115  16   3]
    # E  7 [  0   0   0  25  21   2]
    #    8 [  0   0   0   1   4   0]]
    #         3   4   5   6   7   8
    #               PREDICTED
    #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def get_opt_model(clf, param_grid, X_train, y_train, scoring = 'accuracy'):
    '''
    PARAMETERS:
    clf(classifier): classifier to run GridSearch on
    param_grid(dict): dictionary of parameters to test in GridSearch
    X_train(array): array of predictor variables
    y_train(array): array of response variable
    scoring: score to maximize during GridSearch, default to accuracy

    PRINTS: best training score and best parameter values determined by GridSearch
    RETURNS: fitted model (based on entire train set) with best parameter values
    '''

    best_clf = GridSearchCV(clf, param_grid, scoring = scoring, cv = 5, n_jobs = -1)
    best_clf.fit(X_train, y_train)
    print(best_clf.best_score_)
    print(best_clf.best_params_)
    return best_clf.best_estimator_

def print_confusion_matrix(best_clf, X_test, y_test):
    '''
    PARAMETERS:
    best_clf(classifier): classifier with optimal parameters determined by GridSearch
    X_test(array): array of test set predictor variables
    y_test(array): array of test set response variable

    PRINTS: test accuracy and confusion matrix
    '''

    #predictions made by model
    clf_predicts = best_clf.predict(X_test)
    print(best_clf.score(X_test, y_test))
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

    '''
    testing random forest classifier
    '''
    # rf = RandomForestClassifier()
    # rf_test_params = {'n_estimators': [30, 35, 40, 45, 50], 'max_depth': [15, 20, 25, 30, 35], \
    #             'min_samples_leaf': [1, 5]}
    # best_rf = get_opt_model(rf, rf_test_params, X_train, y_train)
    # print_confusion_matrix(best_rf, X_test, y_test)

    #CV Training Accuracy: 0.686405337781
    #
    # {'n_estimators': 40, 'max_depth': 25, 'min_samples_leaf': 1}
    #
    #Test Accuracy: 0.68
    #     [[  0   0   1   0   0   0]
    # T  4 [  0   0   7   6   0   0]
    # R  5 [  0   0 129  34   1   0]
    # U  6 [  0   0  41 118  10   0]
    # E  7 [  0   0   0  22  25   1]
    #    8 [  0   0   0   0   5   0]]
    #         3   4   5   6   7   8
    #               PREDICTED
    #

    '''
    testing gradient boosting classifier
    '''
    # gb = GradientBoostingClassifier()
    # gb_test_params = {'n_estimators': [100, 500, 750], 'learning_rate': [.01, .05, .1, .2], \
    #                 'min_samples_leaf': [1, 5, 7]}
    # best_gb = get_opt_model(gb, gb_test_params, X_train, y_train)
    # print_confusion_matrix(best_gb, X_test, y_test)

    #CV Training Accuracy: 0.656380316931
    #
    # {'n_estimators': 500, 'learning_rate': 0.05, 'min_samples_leaf': 5}
    #
    #Test Accuracy: 0.65
    #    3 [[ 0   0   1   0   0   0]
    # T  4 [  0   1   8   4   0   0]
    # R  5 [  1   0 123  39   1   0]
    # U  6 [  0   2  33 115  16   3]
    # E  7 [  0   0   0  25  21   2]
    #    8 [  0   0   0   1   4   0]]
    #         3   4   5   6   7   8
    #               PREDICTED
    #

    '''
    SMOTE analysis - Random Forest
    '''
    # smt = SMOTE()
    # rf_syn = RandomForestClassifier()
    #
    # steps = [('smote', smt), ('rf', rf_syn)]
    # pipeline = Pipeline(steps)
    #
    # rf_syn_test_params = {'rf__n_estimators': [35, 40, 45, 50], 'rf__max_depth': [10, 15, 20, 25, 30], \
    #             'rf__min_samples_leaf': [1, 5]}
    #
    # best_syn_rf = get_opt_model(pipeline, rf_syn_test_params, X_train, y_train)
    # print_confusion_matrix(best_syn_rf, X_test, y_test)

    #CV Training Accuracy: 0.633861551293
    #
    # {'rf__max_depth': 20, 'rf__min_samples_leaf': 1, 'rf__n_estimators': 50}
    #
    #Test Accuracy: 0.62
    # [[  0   1   0   0   0   0]
    #  [  1   2   5   4   1   0]
    #  [  4  15 109  31   5   0]
    #  [  0   7  32 102  23   5]
    #  [  0   1   0  11  34   2]
    #  [  0   0   0   0   4   1]]

    '''
    SMOTE analysis - Gradient Boosting
    '''
    smt = SMOTE()
    gb_syn = GradientBoostingClassifier()
    gb_syn_test_params = {'gb__n_estimators': [500, 750, 1000], 'gb__learning_rate': [.01, .05, .1, .2, .5], \
                     'gb__min_samples_leaf': [1, 5, 7]}

    steps2 = [('smote', smt), ('gb', gb_syn)]
    pipeline2 = Pipeline(steps2)

    best_syn_gb = get_opt_model(pipeline2, gb_syn_test_params, X_train, y_train)
    print_confusion_matrix(best_syn_gb, X_test, y_test)

    #CV Train Accuracy: 0.610508757298
    #
    # {'gb__learning_rate': 0.05, 'gb__min_samples_leaf': 7, 'gb__n_estimators': 1000}
    #
    #Test Accuracy: 0.6275
    #
    # [[  0   0   1   0   0   0]
    #  [  1   1   5   5   1   0]
    #  [  3   8 109  41   3   0]
    #  [  0   3  32 111  20   3]
    #  [  0   1   0  14  29   4]
    #  [  0   0   0   1   3   1]]

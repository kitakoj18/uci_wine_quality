import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


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

    #testing random forest classifier
    rf = RandomForestClassifier()
    test_params = {'n_estimators': [15, 20, 25, 30], 'max_depth': [8, 10, 15], \
                'min_samples_leaf': [1, 5]}
    best_rf = GridSearchCV(rf, test_params, scoring = 'accuracy', cv = 5, n_jobs = -1)
    best_rf.fit(X_train_sc, y_train)
    print(best_rf.best_score_)
    print(best_rf.best_params_)

    rf_predicts = best_rf.predict(X_test_sc)
    print(confusion_matrix(y_test, rf_predicts))

    '''
    Accuracy: 0.670558798999
    Best Parameters: {'n_estimators': 25, 'max_depth': 10, 'min_samples_leaf': 1}
    '''

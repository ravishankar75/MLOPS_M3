import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

def train_model():
    
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')
    
    # Define classifier with their respective hyperparameters
    
    model = RandomForestClassifier()
    
    params = {
                'classifier__n_estimators': [ 100, 200],
                'classifier__criterion': ['entropy', 'log_loss'],
                'classifier__max_depth': [10, 20],
                'classifier__min_samples_split': [2, 5]
            }
       
    # Iterate over classifiers
 
    grid_search = GridSearchCV(model, params, scoring='accuracy', cv=5)
    grid_search.fit( X_train, y_train)
    
    # Print best hyperparameters
    print("Best hyperparameters:")
    print(grid_search.best_params_)

    # Print best score
    print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))

if __name__ == "__main__":
    train_model()

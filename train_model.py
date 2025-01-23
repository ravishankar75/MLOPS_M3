import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import mlflow
import warnings
warnings.filterwarnings("ignore")

def train_model():
    
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')
    
    # Define solvers to tune
    classifier__solver = ['newton-cg', 'lbfgs', 'liblinear']

    # Set MLflow experiment name
    mlflow.set_experiment("Logistic Regression Solver Tuning")

    # Start experimenting
    for solver in classifier__solver:
        with mlflow.start_run(run_name=f"Solver: {solver}"):
            try:
                # Train Logistic Regression model
                model = LogisticRegression(solver=solver, random_state=42, max_iter=1000)
                model.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                # Log parameters and metrics
                mlflow.log_param("solver", solver)
                mlflow.log_metric("accuracy", acc)
                
                # Log the dataset file as an artifact
                mlflow.log_artifact("X_train.csv", artifact_path="data")
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                print(f"Solver: {solver} | Accuracy: {acc}")

            except Exception as e:
                print(f"Solver {solver} failed with error: {e}")
                mlflow.log_param("solver", solver)
                mlflow.log_metric("error", 1)


if __name__ == "__main__":
    train_model()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


def prepare_data():
    potable_data  = pd.read_csv('data/water_potability.csv')
    
    # Data Visualization and Exploration 

    print ("Printing first 2 rows to validate data load from csv")
    print (potable_data.head(2))

    print ("Printing information of the data loaded")
    print (potable_data.info())

    # Using the describe method of dataframe to share mean, std deviation & 5 point summary
    print ("Printing description of the data loaded")
    print (potable_data.describe())

    # Using the inull method of dataframe to sum the null, NaN (missing) data by column
    missing_values = potable_data.isnull().sum()
    print("Missing Values:",missing_values )
    
    # Data Pre-processing and cleaning [2 M]
    # To do steps - pre process - NA, Clean data

    # Missing values - print (potable_data.isna().sum())
    # PH = 491, Sulfate =781 , Trihalomethone = 162 
    # Reference - https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html
    # From the plot - KNN Imputer has best MSE outcome

    # ML algorithms benefit from standardization of the data set 
    # Reference 1 - https://scikit-learn.org/stable/modules/preprocessing.html
    # Reference 2 - http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html
    # Considered StandardScalar, MinMaxScalar, RobustScalar . RobustScalar -Selected, support for Outliers
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html

    # Split the data into features and target variable
    X = potable_data.drop(columns =['Potability'])  # Assign features X Array
    Y = potable_data["Potability"]  # Potable Category - Label

    # Reference documentation - https://scikit-learn.org/stable/common_pitfalls.html 
    # To avoid data leakage between Train and test data, split and complete pre-processing

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # address the missing values through KNNImputer from skLearn 
    knn_imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = knn_imputer.fit_transform(X_train)

    # Scaling helps in Most of the data has outliers,
    Robust_scaler = preprocessing.RobustScaler()
    X_train_rscaled = Robust_scaler.fit_transform(X_train_imputed)

    X_train = X_train_rscaled

    X_test = knn_imputer.transform (X_test)
    X_test = Robust_scaler.transform(X_test)

    pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
    
    pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

if __name__ == "__main__":
    prepare_data()

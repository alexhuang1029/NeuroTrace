#This is a helper used for the ML portion of the section. It keeps from having to remove the "ID" column everytime num_cols is used.
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

#import previous helper
from eda_helper import car_df, cat_cols, num_cols
#copy car df
car_df=car_df.copy()
if "ID" in num_cols:
    num_cols.remove("ID")
if "Price" in num_cols:
    num_cols.remove("Price")

def calculate_cramers_v(df, categorical_columns):
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    cramers_corr = pd.DataFrame(index=categorical_columns, columns=categorical_columns, dtype=float)

    for col1 in categorical_columns:
        for col2 in categorical_columns:
            cramers_corr.loc[col1, col2] = cramers_v(df[col1], df[col2])
    
    print("\nCramer's V Correlation for Categorical Features:")
    print(cramers_corr)
    
def encode_categorical_columns(df, categorical_columns):
    """
    Returns:
    pandas.DataFrame: A new DataFrame with categorical columns encoded.
    dict: A dictionary of LabelEncoders used for each categorical column.
    """
    encoded_df = df.copy()
    encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        encoded_df[col] = encoder.fit_transform(encoded_df[col])
        encoders[col] = encoder
    return encoded_df
encoded_df = encode_categorical_columns(car_df, cat_cols)
encoded_df.drop(columns=['Price'], inplace=True)


def rfr_feature_importance(df, target_column):
    """
    Returns:
    pandas.DataFrame: A DataFrame with features and their importance scores.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    #Calculate feature importances using generated Random Forest Regressor Tree Model
    rf = RandomForestRegressor()
    rf.fit(X, y)
    feature_importances = rf.feature_importances_

    # generate df and order to display feature hiearchy 
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df
def anova_test(df, categorical_columns, alpha=0.05):
    for col in categorical_columns:
        categories = df[col].unique()
        category_groups = []

        for category in categories:
            category_data = df[df[col] == category]['Price USD']
            category_groups.append(category_data)

        f_statistic, p_value = stats.f_oneway(*category_groups)

        print(f"ANOVA Result for '{col}':")
        print(f"F-statistic: {f_statistic:.2f}") 
        print(f"P-value: {p_value}")

        if p_value < alpha:
            print(f"At alpha = {alpha}: There is a significant difference between Price and {col}.")
        else:
            print(f"At alpha = {alpha}: There is no significant difference between Price and {col}.")

            
def preprocess_data(X_train, X_test):
    # Normalize a version of the X_train set
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train.copy())
 
    # Standardize a version of the X_train and X_test sets
    X_train_stand = X_train.copy()
    X_test_stand = X_test.copy()
 
    num_cols_to_standardize = ['Model Year', 'Mileage']
 
    for col in num_cols_to_standardize:
        # Fit on training data column
        scale = StandardScaler().fit(X_train_stand[[col]])
 
        # Transform the training data column
        X_train_stand[col] = scale.transform(X_train_stand[[col]])
 
        # Transform the testing data column
        X_test_stand[col] = scale.transform(X_test_stand[[col]])
 
    return X_train_norm, X_train_stand, X_test_stand
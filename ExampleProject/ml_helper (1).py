
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import sklearn.metrics as sm
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn.preprocessing import StandardScaler


car_df = pd.read_csv('Clean Data_pakwheels.csv') #load csv

def preprocess_car_df(car_df):
    # Copy the dataframe
    df_copy = car_df.copy()

    # Convert price column to USD
    df_copy["Price"] = df_copy["Price"] * 0.0035

    # Rename the "Price" column to "Price USD"
    df_copy.rename(columns={"Price": "Price USD"}, inplace=True)

    df_copy.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
    
    df_copy['Mileage'] = df_copy['Mileage'] * 0.621371
    # Calculate Car Age (uncomment and adjust if needed)
    # current_year = datetime.now().year
    # df_copy['Car Age'] = current_year - df_copy['Model Year']
    return df_copy

def drop_irrelevant(car_df):
    # Drop unnecessary columns
    df_copy = car_df.copy()
    
    columns_to_drop = ["ID", "Location", "Engine Type", "Engine Capacity", 
                       "Color", "Assembly", "Registration Status", "Company Name"]
    df_copy.drop(columns=columns_to_drop, inplace=True)

    # Convert mileage column from km to miles
    df_copy['Mileage'] = df_copy['Mileage'] * 0.621371

    return df_copy

def one_hot_encode_dataframe(df, columns):
    # Perform one-hot encoding on specified columns
    df_encoded = pd.get_dummies(df, columns=columns)
    return df_encoded

def train_and_evaluate_linear_regression(linear_df_encoded):
    # Split the data into features and target
    X = linear_df_encoded.drop('Price USD', axis=1)  # Features
    y = linear_df_encoded['Price USD']  # Target

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)

    # Calculate and print evaluation metrics
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2))
    print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2))
    print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))

    # Optionally, return the model and metrics if you need to use them later
    metrics = {
        "MAE": sm.mean_absolute_error(y_test, y_pred),
        "MSE": sm.mean_squared_error(y_test, y_pred),
        "Median AE": sm.median_absolute_error(y_test, y_pred),
        "Explained Variance": sm.explained_variance_score(y_test, y_pred),
        "R2 Score": sm.r2_score(y_test, y_pred)
    }

    return model, metrics, X_test, y_test

def filter_outliers(df, column_name):
    # Calculate Q1 and Q3
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    return filtered_df

def linear_regression_filtered_mileage(df_encoded):
    # Split the data into features and target
    X = df_encoded.drop('Price USD', axis=1)  # Features
    y = df_encoded['Price USD']  # Target

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)

    # Calculate and print evaluation metrics
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2))
    print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2))
    print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))

    # Optionally, return the model and metrics
    metrics = {
        "MAE": sm.mean_absolute_error(y_test, y_pred),
        "MSE": sm.mean_squared_error(y_test, y_pred),
        "Median AE": sm.median_absolute_error(y_test, y_pred),
        "Explained Variance": sm.explained_variance_score(y_test, y_pred),
        "R2 Score": sm.r2_score(y_test, y_pred)
    }

    return model, metrics

def linear_regression_kfold_cv(df_encoded, k=5, random_state=42):
    # Split the data into features and target
    X = df_encoded.drop('Price USD', axis=1)  # Features
    y = df_encoded['Price USD']  # Target

    # Create a KFold object
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Initialize lists to store the results of different metrics
    mae_scores = []
    mse_scores = []
    median_ae_scores = []
    exp_var_scores = []
    r2_scores = []

    # Create a Linear Regression model
    model = LinearRegression()

    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate and store the metrics
        mae_scores.append(sm.mean_absolute_error(y_test, y_pred))
        mse_scores.append(sm.mean_squared_error(y_test, y_pred))
        median_ae_scores.append(sm.median_absolute_error(y_test, y_pred))
        exp_var_scores.append(sm.explained_variance_score(y_test, y_pred))
        r2_scores.append(sm.r2_score(y_test, y_pred))

    # Calculate and return the average of each metric
    avg_metrics = {
        "Average MAE": np.mean(mae_scores),
        "Average MSE": np.mean(mse_scores),
        "Average Median AE": np.mean(median_ae_scores),
        "Average Explained Variance": np.mean(exp_var_scores),
        "Average R2 Score": np.mean(r2_scores)
    }

    return avg_metrics

def ridge_regression_with_outlier_removal(df_filtered_encoded):
    # Split the data into features and target
    X_r = df_filtered_encoded.drop('Price USD', axis=1)  # Features
    y_r = df_filtered_encoded['Price USD']  # Target

    # Splitting the data into training and testing sets
    X_rtrain, X_rtest, y_rtrain, y_rtest = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_rtrain_scaled = scaler.fit_transform(X_rtrain)
    X_rtest_scaled = scaler.transform(X_rtest)

    # Create and train the Ridge regression model
    ridge_model = Ridge(alpha=1.0)  # Adjust alpha as needed
    ridge_model.fit(X_rtrain_scaled, y_rtrain)

    # Predict and evaluate the model
    y_rpred = ridge_model.predict(X_rtest_scaled)

    # Calculate and print evaluation metrics
    metrics = {
        "Mean Absolute Error": sm.mean_absolute_error(y_rtest, y_rpred),
        "Mean Squared Error": sm.mean_squared_error(y_rtest, y_rpred),
        "Median Absolute Error": sm.median_absolute_error(y_rtest, y_rpred),
        "Explained Variance Score": sm.explained_variance_score(y_rtest, y_rpred),
        "R2 Score": sm.r2_score(y_rtest, y_rpred)
    }

    return ridge_model, metrics, X_rtrain_scaled, X_rtest_scaled, y_rtrain, y_rtest


def ridge_cv_hyperparameter_tuning(X_train, y_train, X_test, y_test, alpha_values=None):
    # Define a range of alpha values for testing if not provided
    if alpha_values is None:
        alpha_values = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

    # Create and train the Ridge regression model with built-in cross-validation
    ridge_cv_model = RidgeCV(alphas=alpha_values, store_cv_values=True)
    ridge_cv_model.fit(X_train, y_train)

    # Optimal alpha value
    optimal_alpha = ridge_cv_model.alpha_
    print("Optimal alpha value:", optimal_alpha)

    # Predict and evaluate the model
    y_pred = ridge_cv_model.predict(X_test)

    # Calculate and print evaluation metrics
    metrics = {
        "Mean Absolute Error": sm.mean_absolute_error(y_test, y_pred),
        "Mean Squared Error": sm.mean_squared_error(y_test, y_pred),
        "Median Absolute Error": sm.median_absolute_error(y_test, y_pred),
        "Explained Variance Score": sm.explained_variance_score(y_test, y_pred),
        "R2 Score": sm.r2_score(y_test, y_pred)
    }

    return ridge_cv_model, metrics

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import sklearn.metrics as sm

def lasso_regression_with_outlier_removal(df_filtered_encoded):
    # Split the data into features and target
    X_l = df_filtered_encoded.drop('Price USD', axis=1)  # Features
    y_l = df_filtered_encoded['Price USD']  # Target

    # Splitting the data into training and testing sets
    X_ltrain, X_ltest, y_ltrain, y_ltest = train_test_split(X_l, y_l, test_size=0.2, random_state=42)

    # Create and train the Lasso regression model
    lasso_model = Lasso()
    lasso_model.fit(X_ltrain, y_ltrain)

    # Predict and evaluate the model
    y_lpred = lasso_model.predict(X_ltest)

    # Calculate and print evaluation metrics
    metrics = {
        "Mean Absolute Error": sm.mean_absolute_error(y_ltest, y_lpred),
        "Mean Squared Error": sm.mean_squared_error(y_ltest, y_lpred),
        "Median Absolute Error": sm.median_absolute_error(y_ltest, y_lpred),
        "Explained Variance Score": sm.explained_variance_score(y_ltest, y_lpred),
        "R2 Score": sm.r2_score(y_ltest, y_lpred)
    }

    for metric, value in metrics.items():
        print(f"{metric} =", round(value, 2))

    return lasso_model, metrics, X_ltrain, X_ltest, y_ltrain, y_ltest

def plot_model_performance(df, metric_names):
    """
    Plots the performance of different regression models based on specified metrics.
    
    Parameters:
    df (DataFrame): A DataFrame containing the model names and their metrics.
    metric_names (list): List of metric column names in the DataFrame.
    """
    # Check if all metric names are in DataFrame
    for metric in metric_names:
        if metric not in df.columns:
            raise ValueError(f"{metric} not found in DataFrame")

    # Number of models and metrics
    n_models = len(df)
    n_metrics = len(metric_names)

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35 / n_metrics  # Adjust bar width
    index = np.arange(n_models)

    for i, metric in enumerate(metric_names):
        ax.bar(index + i * bar_width, df[metric], bar_width, label=metric)

    ax.set_xlabel('Model Type')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Regression Models')
    ax.set_xticks(index + bar_width * n_metrics / 2)
    ax.set_xticklabels(df['Model'])
    ax.legend()

    plt.xticks(rotation=45)
    plt.show()
    
import matplotlib.pyplot as plt

def plot_residuals(y_true, predictions, model_names):
    
    plt.figure(figsize=(12, 8))

    for i, model in enumerate(model_names):
        if model in predictions:
            residuals = y_true - predictions[model]
            plt.subplot(1, len(model_names), i+1)
            plt.scatter(predictions[model], residuals, alpha=0.5)
            plt.title(f"{model} Residuals")
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.axhline(y=0, color='r', linestyle='--')
        else:
            print(f"Warning: {model} not found in predictions.")

    plt.tight_layout()
    plt.show()
    


def train_random_forest(df_filtered_encoded, target_column='Price USD', test_size=0.2, random_state=42):
    
    # Splitting the data into features and target
    X = df_filtered_encoded.drop(target_column, axis=1)
    y = df_filtered_encoded[target_column]

    # Splitting the data into training and testing sets
    X_rftrain, X_rftest, y_rftrain, y_rftest = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Creating and training the Random Forest model
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X_rftrain, y_rftrain)

    # Predicting and evaluating the model
    y_rfpred = model.predict(X_rftest)

    # Calculating evaluation metrics
    metrics = {
        "Mean Absolute Error": mean_absolute_error(y_rftest, y_rfpred),
        "Mean Squared Error": mean_squared_error(y_rftest, y_rfpred),
        "Median Absolute Error": median_absolute_error(y_rftest, y_rfpred),
        "Explained Variance Score": explained_variance_score(y_rftest, y_rfpred),
        "R2 Score": r2_score(y_rftest, y_rfpred)
    }

    # Printing evaluation metrics
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name} = {round(metric_value, 2)}")

    return model, metrics, X_rftrain, X_rftest, y_rftrain, y_rftest

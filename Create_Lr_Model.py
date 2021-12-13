import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('./data/video_game_sales.csv')

enc = OneHotEncoder(categories='auto', handle_unknown='ignore')
feature_arr = enc.fit_transform(df[['Genre', 'Platform']]).toarray()
feature_labels = enc.get_feature_names_out()
feature_labels = np.array(feature_labels, dtype=object).ravel()
features = pd.DataFrame(feature_arr, columns=feature_labels)

df = df.join(features).dropna()

# Break down into X inputs and Y output, drop columns we don't care about
x = df.drop(['Global_Sales', 'Platform', 'Genre', 'Publisher', 'Name'], axis=1)

x = x
y = df.Global_Sales

# 20% of the data set will go to the test size, 80% to training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Perform Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)

# Make predictions on training set
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)


def show_real_vs_predicted():
    plt.figure(figsize=(10, 10))
    plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)
    z = np.polyfit(y_train, y_lr_train_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_train, p(y_train), "#F8766D")
    plt.ylabel('Predicted Sales')
    plt.xlabel('Actual Sales')

    return plt


def get_linear_regression_metrics():
    # Build metrics for model performance
    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)

    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)

    lr_results = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
    lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

    return lr_results


def run_random_forest():
    # Random Forest Algorithm
    rf = RandomForestRegressor(max_depth=2, random_state=42)
    rf.fit(x_train, y_train)

    y_rf_train_pred = rf.predict(x_train)
    y_rf_test_pred = rf.predict(x_test)

    # Random Fores Metrics
    rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
    rf_train_r2 = r2_score(y_train, y_rf_train_pred)

    rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
    rf_test_r2 = r2_score(y_test, y_rf_test_pred)

    rf_results = pd.DataFrame(['Random forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
    rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

    return rf_results


def run_knearest_neighbor():
    # Nearest Neighbor Algorithm
    kn = KNeighborsRegressor(n_neighbors=2)
    kn.fit(x_train, y_train)

    y_kn_train_pred = kn.predict(x_train)
    y_kn_test_pred = kn.predict(x_test)

    # Nearest Neighbor Metrics
    kn_train_mse = mean_squared_error(y_train, y_kn_train_pred)
    kn_train_r2 = r2_score(y_train, y_kn_train_pred)

    kn_test_mse = mean_squared_error(y_test, y_kn_test_pred)
    kn_test_r2 = r2_score(y_test, y_kn_test_pred)

    kn_results = pd.DataFrame(['K Nearest Neighbor', kn_train_mse, kn_train_r2, kn_test_mse, kn_test_r2]).transpose()
    kn_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

    return kn_results


def get_data_frame():
    return df
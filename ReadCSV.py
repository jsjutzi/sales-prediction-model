import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

prep = preprocessing.LabelEncoder()


df = pd.read_csv('./data/video_game_sales.csv')

enc = OneHotEncoder(categories='auto', handle_unknown='ignore')
feature_arr = enc.fit_transform(df[['Genre', 'Platform']]).toarray()
feature_labels = enc.get_feature_names_out()
feature_labels = np.array(feature_labels, dtype=object).ravel()
features = pd.DataFrame(feature_arr, columns=feature_labels)

df = df.join(features).dropna()

# Clean and prepare data


# Break down into X inputs and Y output
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

# Build metrics for model performance
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

lr_results = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']



def print_thing():
    print(lr_results)








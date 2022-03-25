import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv('train.csv')

df = df.drop('time', axis = 1)
df = df.drop('row_id', axis = 1)

df = pd.get_dummies(df, drop_first=True)

X = pd.read_csv('X.csv', index_col=[0])
y = pd.read_csv('y.csv', index_col=[0])

print(df.shape)

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit()

for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

rang = y_train.values.max() - y_train.values.min()

from sklearn.linear_model import LinearRegression
from sklearn.ensemble      import RandomForestRegressor
from xgboost               import XGBRegressor
from sklearn.svm import SVR

regressors = {
  "Linear Regression": LinearRegression(),
  "Random Forest": RandomForestRegressor(max_depth=10, random_state=0),
  "XGBoost": XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5, gamma=0, subsample=0.8),
  "Support Vector Regression": SVR(kernel='poly', C=1.5, degree=3, epsilon=0.2),
}

from sklearn import metrics

results = pd.DataFrame({'Model': [], 'MSE': [], 'MAE': [], 'r2_score': [], " % error": []})

for model_name, model in regressors.items():
    model.fit(X, y)
    pred = model.predict(X_test)
    
    results = results.append({"Model": model_name,
                            "MSE": metrics.mean_squared_error(y_test, pred),
                            "MAE": metrics.mean_absolute_error(y_test, pred),
                            "r2_score": metrics.r2_score(y_test, pred),
                            " % error": metrics.mean_squared_error(y_test, pred) / rang},
                            ignore_index=True)


results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['MSE', 'MAE'], vmin=0, vmax=100, color='#5fba7d')

print(results_ord)
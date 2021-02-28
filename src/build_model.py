# %%
import math
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

plt.style.use('ggplot')

# TODO
# ^^ - exponential smooth to reduce outliers in data
# ^^ - standard vs normalize scaling
# ^^ - etc.

# %%
df = pd.read_csv('../tickers_with_features/GME.csv',
                 index_col='Date', parse_dates=True)
df.shape

# %%
# Shift 1 day backwards so the data from the 'current' tries to predict
# tomorrows output
df['y'] = df['Adj Close'].shift(-5)
df.dropna(inplace=True)

# %%
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# %%
X_train.tail()

# %%
X_test.head()

# %%
X_train_not_scaled = X_train.copy()
X_test_not_scaled = X_test.copy()

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train[X_train.columns] = scaler.transform(X_train[X_train.columns])
X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])

scaler.fit(y_train)
y_train[y_train.columns] = scaler.transform(y_train[y_train.columns])
y_test[y_test.columns] = scaler.transform(y_test[y_test.columns])

# y nie potrzebuje scalowania, bo to jest etykieta

# %%
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
y_pred = model.predict(X_test)

# %%

# %%
# evaluate predictions
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Deviation:', math.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2:', metrics.r2_score(y_test, y_pred))
# print('AIC:', regr.au)
# print('BIC:', metrics.b(y_test, y_pred))
# print('AUC:', metrics.a(y_test, y_pred))

# %%
# model = tree.DecisionTreeRegressor()
# model = linear_model.Lasso()
# model.fit(X_train, y_train)
# # evaluate the model
# y_pred = model.predict(X_test)
# # evaluate predictions
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Root Mean Squared Deviation:', math.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('R2:', metrics.r2_score(y_test, y_pred))



# %%
fig, ax = plt.subplots()
ax.plot(df['Adj Close'], lw=0.5, alpha=1)
ax.plot(y_test, lw=0.5, c='w', alpha=1)
y_pred_with_date = pd.DataFrame(y_pred, y_test.index)
ax.plot(y_pred_with_date, lw=0.5, c='r', alpha=1)
plt.legend(['Adj Close', 'Dane Prognozowane'])
fig.autofmt_xdate()
plt.show()

# %%

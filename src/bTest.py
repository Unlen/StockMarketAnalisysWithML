# %%
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from sklearn import linear_model, metrics, tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor

plt.style.use('unlen')

tickers_dir = './tickers_with_features/'
output_dir = './output/'
# TODO
# ^^ - exponential smooth to reduce outliers in data
# ^^ - standard vs normalize scaling
# ^^ - etc.


# %%
def split_df_for_train_test_data(df, n_days):
    # Create labels
    # Shift n days backwards so the data from the 'current' tries to predict
    # future output.
    df['y'] = df['Adj Close'].shift(-n_days)

    # Afetr a shift, there is n days without label.
    df.dropna(inplace=True)

    # Separate features and labels.
    X = df.iloc[:, :-1]  # features
    y = df.iloc[:, -1:]  # labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


def evaluate_model(ticker, y_test, y_pred, n_days_shift):
    with open(output_dir + ticker + '/' + str(n_days_shift) + '_days_metrics.txt', 'w') as out:
        out.write('Mean Absolute Error: %.6f' %
                  metrics.mean_absolute_error(y_test, y_pred) + '\n')
        out.write('Root Mean Squared Deviation: %.6f' % math.sqrt(
            metrics.mean_squared_error(y_test, y_pred)) + '\n')
        out.write('R2: %.6f' % metrics.r2_score(y_test, y_pred) + '\n')

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Root Mean Squared Deviation:', math.sqrt(
        metrics.mean_squared_error(y_test, y_pred)))
    print('R2:', metrics.r2_score(y_test, y_pred))


def get_sample_from(df, pos):
    return np.array(df.iloc[pos, :].values.tolist()).reshape(1, -1)


def predict_future_prices_for(n_days, X_train, X_test, model):
    df_n_days_predictions = pd.DataFrame()

    for n_day in reversed(range(-1, -n_days - 1, -1)):
        sample = get_sample_from(X_train, n_day)
        prediction = model.predict(sample)

        next_bday = X_test.index[n_day + n_days]
        df_pred = pd.DataFrame(prediction, [next_bday], ['Adj Close'])
        df_n_days_predictions = df_n_days_predictions.append(df_pred)

    return df_n_days_predictions


def read_scaled_data_from_ticker(ticker, n_days_shift):
    df = pd.read_csv(tickers_dir + ticker + '.csv',
                     index_col='Date', parse_dates=True)

    X_train, X_test, y_train, y_test = split_df_for_train_test_data(
        df, n_days_shift)

    # It is important to fit the scaler after train/test split, so
    # the test data wont influence training data in any way.
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(X_train)
    # Train and Test data have to be scaled on the same scale.
    X_train[X_train.columns] = scaler_X.transform(X_train[X_train.columns])
    X_test[X_test.columns] = scaler_X.transform(X_test[X_test.columns])

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(y_train)
    y_train[y_train.columns] = scaler_y.transform(y_train[y_train.columns])
    y_test[y_test.columns] = scaler_y.transform(y_test[y_test.columns])

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


def save_n_days_predictions_to_txt(ticker, df_actual_prices, df_predicted_prices, n_days_shift):
    with open(output_dir + ticker + '/' + str(n_days_shift) + '_days_prediction.txt', 'w') as out:
        actual = df_actual_prices['Adj Close']
        pred = df_predicted_prices['Adj Close']
        df = pd.concat([actual, pred],
                       axis=1, keys=['Cena Aktualna', 'Cena Prognozowana'])
        df.to_string(out, float_format='%.2f', justify='center')


def save_predictions_to_txt(ticker, X_test, y_pred_shifted, n_days_shift):
    with open(output_dir + ticker + '/' + str(n_days_shift) + '_days_prediction_pred_vs_actual.txt', 'w') as out:
        actual = X_test['Adj Close']
        pred = y_pred_shifted['y']
        df = pd.concat([actual, pred],
                       axis=1, keys=['Cena Aktualna', 'Cena Prognozowana'])
        df.to_string(out, float_format='%.2f', justify='center')


def save_n_days_diagram(ticker, X_train, df_actual_prices, df_predicted_prices, n_days_shift):
    fig, ax = plt.subplots()
    ax.plot(X_train.iloc[-30: -1, :]['Adj Close'], lw=0.5, c='w', alpha=1)
    ax.plot(df_actual_prices['Adj Close'])
    ax.plot(df_predicted_prices)
    fig.autofmt_xdate()
    plt.title('Predykcje ' + str(n_days_shift) +
              ' dni wprzód [' + ticker + ']')
    plt.ylabel('Cena')
    plt.xlabel('Data')
    plt.legend(['Cena Historyczna', 'Cena Aktualna', 'Cena Prognozowana'])
    fig.savefig(output_dir+ticker+'/' +
                str(n_days_shift) + '_days_prediction.svg')


def save_predictions_vs_actual_diagram(ticker, X_test, y_pred_shifted, n_days_shift):
    fig, ax = plt.subplots()
    ax.plot(X_test['Adj Close'], lw=0.5, c='w', alpha=1)
    ax.plot(y_pred_shifted['y'], lw=0.5, c='r', alpha=1)
    fig.autofmt_xdate()
    plt.title('Predykcje ' + str(n_days_shift) + ' dni wprzód [PFE]')
    plt.ylabel('Cena')
    plt.xlabel('Data')
    plt.legend(['Cena Aktualna', 'Cena Prognozowana'])
    fig.savefig(output_dir+ticker+'/' + str(n_days_shift) +
                '_days_prediction_pred_vs_actual.svg')


def run_ml_for(tickers, n_days_shift, market):
    # reg = linear_model.LinearRegression()
    reg = linear_model.RidgeCV()
    # reg = linear_model.LinearRegression()
    # reg = tree.DecisionTreeRegressor()
    # reg = RandomForestRegressor()
    # reg = KNeighborsRegressor(n_neighbors=20, metric='euclidean')
    # estimators = [('linear', linear_model.LinearRegression()),
    #               ('tree', tree.DecisionTreeRegressor()),
    #               ('forest', RandomForestRegressor()),
    #               ('ridge', linear_model.RidgeCV()),
    #               ('lasso', linear_model.LassoCV(random_state=42, tol=1e-3)),
    #               ('knr', KNeighborsRegressor(n_neighbors=20,
    #                                           metric='euclidean'))]

    # estimators = [('linear', linear_model.LinearRegression()),
    #             ('tree', tree.DecisionTreeRegressor()),
    #             ('ridge', linear_model.RidgeCV())]
    #             # ('forest', RandomForestRegressor()),
    # final_estimator = GradientBoostingRegressor()
    # reg = StackingRegressor(
    #     estimators=estimators,
    #     final_estimator=final_estimator)

    # Build model on all of the data.
    for ticker in tickers:
        X_train, X_test, y_train, y_test, scaler_X, scaler_y = read_scaled_data_from_ticker(
            ticker, n_days_shift)
        reg.fit(X_train, y_train)

    # Evaluate model for each ticker.
    for ticker in tickers:
        X_train, X_test, y_train, y_test, scaler_X, scaler_y = read_scaled_data_from_ticker(
            ticker, n_days_shift)

        y_pred = reg.predict(X_test)
        evaluate_model(ticker, y_test, y_pred, n_days_shift)

        y_pred_labels = y_test.index
        y_pred = pd.DataFrame(y_pred, y_pred_labels, y_test.columns)
        y_pred[y_pred.columns] = scaler_y.inverse_transform(y_pred[y_pred.columns])

        df_predicted_prices = predict_future_prices_for(
            n_days_shift, X_train, X_test, reg)
        df_predicted_prices[df_predicted_prices.columns] = scaler_y.inverse_transform(
            df_predicted_prices[df_predicted_prices.columns])

        # Reverse scale so the data will be in its normal scale for producing output
        X_train[X_train.columns] = scaler_X.inverse_transform(
            X_train[X_train.columns])
        X_test[X_test.columns] = scaler_X.inverse_transform(
            X_test[X_test.columns])
        y_train[y_train.columns] = scaler_y.inverse_transform(
            y_train[y_train.columns])
        y_test[y_test.columns] = scaler_y.inverse_transform(
            y_test[y_test.columns])

        df_actual_prices = X_test.iloc[0: n_days_shift, :]

        try:
            market_calendar = mcal.get_calendar(market)
        except Exception:
            print('Market calendar not supported! market=%s' % market)
            exit(1)

        y_pred_shifted = y_pred.copy()
        custom_business_day = pd.offsets.CustomBusinessDay(
            n_days_shift, calendar=market_calendar.holidays().calendar)
        # y_pred_shifted.index = y_pred_shifted.index + custom_business_day

        save_n_days_predictions_to_txt(
            ticker, df_actual_prices, df_predicted_prices, n_days_shift)
        save_predictions_to_txt(ticker, X_test, y_pred_shifted, n_days_shift)
        save_n_days_diagram(ticker, X_train, df_actual_prices, df_predicted_prices, n_days_shift)
        save_predictions_vs_actual_diagram(ticker, X_test, y_pred_shifted, n_days_shift)


# %%


# %%

#//! 1. Więcej danych dla podobnych spolek 1/10/50
#//! 2. Scylatory dla swingowych spolek - czy polepszaja czy pogarszaja
#//! 3. 


# TODO opisać op nauce na wszystkich danych jak już się wybierze ostateczny model
# TODO opisać, że rozjazd w datach może wystąpić ze względy na wolne dni w
#      święta, które nie są obrane pod uwagę w tym programie

# %%

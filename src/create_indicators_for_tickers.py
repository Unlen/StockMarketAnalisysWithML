# %%
import glob
import os
from pathlib import Path

import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator, StochasticOscillator
from ta.others import (CumulativeReturnIndicator, DailyLogReturnIndicator,
                       DailyReturnIndicator)
from ta.trend import MACD, EMAIndicator, SMAIndicator, WMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import (AccDistIndexIndicator, ChaikinMoneyFlowIndicator,
                       OnBalanceVolumeIndicator)


def add_trend_indicators(df, adj_close):
    df['SMA50'] = SMAIndicator(adj_close, window=50).sma_indicator()
    df['SMA200'] = SMAIndicator(adj_close, window=200).sma_indicator()
    df['WMA40'] = WMAIndicator(adj_close, window=40).wma()
    df['WMA150'] = WMAIndicator(adj_close, window=150).wma()
    # df['EMA30'] = EMAIndicator(adj_close, window=30).ema_indicator()
    # df['EMA100'] = EMAIndicator(adj_close, window=100).ema_indicator()
    # indicator = MACD(adj_close, window_fast=8, window_slow=24, window_sign=9)
    # df['MACD'] = indicator.macd()
    # df['MACD_diff'] = indicator.macd_diff()
    # df['MACD_signal'] = indicator.macd_signal()


def add_volume_indicators(df, high, low, adj_close, volume):
    df['ADI'] = AccDistIndexIndicator(
        high, low, adj_close, volume).acc_dist_index()
    df['CMF'] = ChaikinMoneyFlowIndicator(
        high, low, adj_close, volume).chaikin_money_flow()
    df['OBV'] = OnBalanceVolumeIndicator(adj_close, volume).on_balance_volume()


def add_volatility_indicators(df, high, low, adj_close):
    df['ATR'] = AverageTrueRange(high, low, adj_close).average_true_range()
    indicator = BollingerBands(adj_close)
    df['BB_hb'] = indicator.bollinger_hband()
    df['BB_hbi'] = indicator.bollinger_hband_indicator()
    df['BB_lb'] = indicator.bollinger_lband()
    df['BB_lbi'] = indicator.bollinger_lband_indicator()
    df['BB_m'] = indicator.bollinger_mavg()
    df['BB_pb'] = indicator.bollinger_pband()
    df['BB_wb'] = indicator.bollinger_wband()


def add_momentum_indicators(df, high, low, adj_close):
    df['ROC'] = ROCIndicator(adj_close).roc()
    df['RSI'] = RSIIndicator(adj_close).rsi()
    indicator = StochasticOscillator(high, low, adj_close)
    df['SO'] = indicator.stoch()
    df['SO_sig'] = indicator.stoch_signal()


def add_return_indicators(df, adj_close):
    df['CR'] = CumulativeReturnIndicator(adj_close).cumulative_return()
    df['DLR'] = DailyLogReturnIndicator(adj_close).daily_log_return()
    df['DR'] = DailyReturnIndicator(adj_close).daily_return()


def drop_not_needed_basic_indicators_from(df):
    df.drop('Open', axis=1, inplace=True)
    df.drop('Close', axis=1, inplace=True)


def create_indicators_for(df, high, low, adj_close, volume):
    df_ret = df.copy()
    # add_trend_indicators(df_ret, adj_close)
    # add_volume_indicators(df_ret, high, low, adj_close, volume)
    # add_volatility_indicators(df_ret, high, low, adj_close)
    # add_momentum_indicators(df_ret, high, low, adj_close)
    # add_return_indicators(df_ret, adj_close)
    return df_ret


def drop_na_rows_from(df):
    last_index_of_null_value = df.isnull().sum().max()
    df.drop(df.index[0:last_index_of_null_value], inplace=True)


def save_df_to_csv(ticker, df):
    name = Path(ticker).name
    df.to_csv(os.path.join('tickers_with_features', name))


def create_indicators():
    tickers = glob.glob(os.path.join('tickers', '*.csv'))

    for ticker in tickers:
        df = pd.read_csv(ticker, index_col=0, parse_dates=True)
        drop_not_needed_basic_indicators_from(df)
        df = create_indicators_for(
            df, df['High'], df['Low'], df['Adj Close'], df['Volume'])
        drop_na_rows_from(df)
        save_df_to_csv(ticker, df)


# TODO opisać, że dane muszą być z conajmniej 5 lat? bo tak jest zrobione MA
# TODO Założenie, że pobrane dane z yahoo finance są poprawne.
# TODO Opisać dropna ?
# TODO dimensionality reduction

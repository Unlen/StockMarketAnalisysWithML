from datetime import datetime
import pandas_datareader as pdr
import time


# TODO napisać w pracy że zdecydowałem się na późniejszą date by szybciej budował się model
#               i by model nie był zaśmiecony za starymi danymi
def download_tickers_from_yahoofinance(tickers):
    start_date = '2010-01-01'
    todays_date = datetime.today().strftime('%Y-%m-%d')
    for ticker in tickers:
        df = pdr.DataReader(ticker, data_source='yahoo', start=start_date, end=todays_date)
        df.to_csv('tickers/' + ticker + '.csv')
        time.sleep(1)  # To not flood the provider

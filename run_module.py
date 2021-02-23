# %%
from src.parse_args import parse_args, read_args
from src.get_tickers_from_yahoo_finance import download_tickers_from_yahoofinance
from src.create_indicators_for_tickers import create_indicators

# TODO Logger for information input
# TODO Add docs to each fuction

if __name__ == '__main__':
    args = read_args()
    args = parse_args(args)

    if args.no_input is False:
        print(' > Downloading %d tickers..' % len(args.tickers))
        download_tickers_from_yahoofinance(args.tickers)
    else:
        print(' > No input provided. Fetching from "./tickers/*"..')

    print(' > Adding indicators for tickers..\n   * Will save to "./tickers_with_features".')
    create_indicators()


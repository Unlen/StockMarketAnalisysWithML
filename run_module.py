# %%
from pathlib import Path

from src.parse_args import parse_args, read_args
from src.get_tickers_from_yahoo_finance import download_tickers_from_yahoofinance
from src.create_indicators_for_tickers import create_indicators
from src.bTest import run_ml_for
import PyQt5
import os

required_direcotries = ['tickers', 'tickers_with_features', 'output']


# dirname = os.path.dirname(PyQt5.__file__)
# plugin_path = os.path.join(dirname, 'plugins', 'platforms')
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


def create_required_directories(required_direcotries, prefix=''):
    for directory in required_direcotries:
        Path(prefix + directory).mkdir(parents=True, exist_ok=True)


# %%
if __name__ == '__main__':
    args = read_args()
    args = parse_args(args)

    create_required_directories(required_direcotries)
    create_required_directories(args.tickers, 'output/')

    if args.no_download is False:
        print(' > Downloading %d tickers..' % len(args.tickers))
        download_tickers_from_yahoofinance(args.tickers)
    else:
        print(' > Skipping downloading. Fetching from "./tickers/*"..')

    if args.no_indicators is False:
        print(' > Adding indicators for tickers..\n   * Will save to "./tickers_with_features/*".')
        create_indicators()
    else:
        print(' > Skipping addition of indicators. Fetching from "./tickers_with_features/*"..')

    print(' > Running ML for tickers..\n   * Will save output to "./output/*".')
    run_ml_for(args.tickers, args.N, args.M)


# python run_module.py 5 NYSE -t PFE --no-download --no-indicators

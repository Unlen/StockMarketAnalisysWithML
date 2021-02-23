import argparse
from argparse import RawTextHelpFormatter
import sys


def read_args():
    program_description =\
        """
        Z
        Q
        A
        D
        """

    parser = argparse.ArgumentParser(description=program_description,
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('-N', type=int, nargs=1, metavar='days', default=5,
                        help='an integer number represent N days that ML model will try to predict the price.')

    # no donwload
    ticker_group = parser.add_mutually_exclusive_group()
    ticker_group.add_argument('-t', '--tickers', nargs='+', metavar='', type=str,
                              help='list of stock companies separated by a white character.')
    ticker_group.add_argument('-tf', '--tickers_file', nargs=1, metavar='path', type=argparse.FileType('r', encoding='utf-8'),
                              default='input_tickers.txt',
                              help='path to file containing stock companies separated by a whitespace character.\n'
                              '* default is \'input_tickers.txt\' in project directory.')
    ticker_group.add_argument('--no-input', action='store_true', default=False,
                              help='to not download tickers if they are already present in "./tickers".')

    # parser.add_argument('datapath', type=pathlib.Path)
    # parser.add_argument('--outdir', nargs='?', type=argparse.FileType('w', encoding='utf-8'),
    #                     default='o.txt')

    args = parser.parse_args()
    return args


def parse_args(args):
    if args.N < 1:
        print("Wrong number of days provided! N=%s" % args.N)
        sys.exit()

    if args.tickers is None:
        file_str = args.tickers_file.read()
        args.tickers = file_str.split()
        if len(args.tickers) == 0:
            print("No stock companies were provided!")
            sys.exit()

    return args


# def validate_input_data():
#     if not N.isnumeric():
#         print("Wrong number of days provided! N=%s" % N)
#         sys.exit()

#     if len(tickers) == 0:
#         print("No stock companies were provided!")
#         sys.exit()

# %%

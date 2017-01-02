"""
$ python cost_basis.py file_path.csv
input file generation:
1) get a data from transaction view in Vanguard. Save as HTML the pages of many years of Vanguard or cut and paste directly the table into a vi file.
2) The program does:
   1) clean up the data (iterative)
   2) Classifies each lot as LT or ST and computes the net for each lot based on the today's price.
   3) Produces a sorted increasing sort of the sale of each lot based LT, ST and fund with the cumulative amount and the cumulative proceeds.
   This is to help decide what to sell at the end of the year. The resulting file is saved in /tmp/g_df.csv
3) computes IRR for each fund and all the funds together assuming all was sold at today's process. Saves file in /tmp/irr_all.csv

MANUAL CLEAN UP:
- remove non-ASCII '- ' from file in vi
- remove transactions that revert like:
        7/29/2015,REIT Index Fund Adm,Buy,-225.693,$110.77,"-$1,000.00"
        7/29/2015,Health Care Fund Adm,Buy,-251.357,$99.46,"-$1000.00"
        7/27/2015,REIT Index Fund Adm,Buy,225.693,$110.77,"$1,000.00"
        7/27/2015,Health Care Fund Adm,Buy,251.357,$99.46,"$1,000.00"
"""


# TODO support for generic column names: buy column names, date column name, sell column names, transaction type column name, fund column name, shares column name, transaction amount column namekjj

import sys
import os
import pandas as pd
import numpy as np
import json
from yahoo_finance import Share
import locale   # to convert numbers like $2,500.35 to 2500.35

try:
   FILE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
   FILE_DIR = '/Users/josep/Code/Financial'
PROJ_DIR = os.path.dirname(FILE_DIR)
CODE_DIR = os.path.dirname(PROJ_DIR)
sys.path.insert(0, PROJ_DIR)
sys.path.insert(0, CODE_DIR)

from Utilities import pandas_utils as pd_ut
from Utilities import time_utils as tm_ut


def tax_status(row, today, date_format):
    if row['Shares'] > 0.0:
        b_date = row['Date']
        return 'LT' if tm_ut.days_diff(b_date, today, date_format=date_format) > 365 else 'ST'
    else:
        return 'NA'


def lot_gain(row, sale_prices):
    fund = row['Fund']
    sale_price = sale_prices[fund]
    shares = row['Shares']
    return shares * sale_price - row['Amount'] if shares > 0 else np.nan


def lot_amount(row, sale_prices):
    fund = row['Fund']
    sale_price = sale_prices[fund]
    shares = row['Shares']
    return shares * sale_price if shares > 0 else np.nan


def fund_smry(a_df):
    a_df.sort_values(by='lot_gain', ascending=True, inplace=True)
    a_df['cum_gain'] = a_df['lot_gain'].cumsum()
    return a_df


def set_price(tkr):
    p = Share(tkr).get_prev_close()
    return 1.0 if p is None else float(p)

THRESHOLD = 1e-8
NUM_ITER = 100


def irr_iteration(p, d, g1=-1.0, g2=1.0):
    """
    :param p: payments: purchases are negative and sales positive. Also the last entry must be a sale of all we have
    :param d: time differences till today in years (fraction)
    :param g1:
    :param g2:
    :return:
    """
    n, npv, g, string = 0, 2.0 * THRESHOLD, np.nan, ''
    while np.abs(npv) > THRESHOLD and n < NUM_ITER:
        g = (g1 + g2) / 2.0
        npv = np.sum(p / np.power(1.0 + g, -d))
        string += 'step %d: npv = %10.10f irr: %3.5f\n' % (n, npv, g)
        n += 1
        g1 = g if npv > 0 else g1
        g2 = g if npv < 0 else g2
    if n < NUM_ITER and np.abs(npv) <= THRESHOLD:
        # print 'convergence'
        return g
    else:
        print string
        print 'no convergence'
        return np.nan


def get_irr(a_df, t_date, d_fmt, today_vals=None):
    # amt invested in (negative)
    buy = a_df[a_df['Transaction Type'] == 'Buy']
    buy_shares = buy['Shares']
    buy_vals = -buy['Amount'].values
    buy_times = buy['Date'].apply(lambda x: tm_ut.days_diff(x, t_date, date_format=d_fmt) / 365.0).values

    # amt received already (positive)
    sell = a_df[a_df['Transaction Type'] == 'Sell']
    sell_shares = -sell['Shares']
    sell_vals = -sell['Amount'].values
    sell_times = sell['Date'].apply(lambda x: tm_ut.days_diff(x, t_date, date_format=d_fmt) / 365.0).values

    # Div and CG
    other = a_df[~a_df['Transaction Type'].isin(['Buy', 'Sell'])]
    other_shares = other['Shares']

    if today_vals is None:
        shares_today = buy_shares.sum() + other_shares.sum() - sell_shares.sum()  # shares available today
        today_vals = shares_today * a_df.ix[a_df.index[0], 'Today Price']
    else:
        shares_today = np.nan

    times_arr = np.append(buy_times, np.append(sell_times, 0))
    vals_arr = np.append(buy_vals, np.append(sell_vals, today_vals))

    total_in = buy_vals.sum()
    total_out = sell_vals.sum() + today_vals
    irr = np.round(100.0 * irr_iteration(vals_arr, times_arr), 2)
    return pd.DataFrame({'IRR%': [irr], 'Return%': [np.round((total_out + total_in) / (-total_in), 2)], 'Today$': ['$' + str(np.round(today_vals, 2))], 'TodaySh': [np.round(shares_today, 2)]})

if __name__ == '__main__':
    # f_cfg = '/Users/josep/Code/Financial/config/cost_basis_cfg.json'
    cwd = os.getcwd() + '/'
    h_dir = os.path.expanduser('~') + '/'
    if len(sys.argv) == 2:
        f_cfg = cwd + sys.argv[1]
    else:
        print 'invalid arguments: ' + str(sys.argv)
        print('ERROR')
        sys.exit(0)

    # read cfg file
    with open(f_cfg, 'r') as fp:
        d_cfg = json.load(fp)

    f_name = d_cfg['in_file']
    tickers = d_cfg['tickers']                       # list of fund in scope with tickers
    date_format = '%m/%d/%Y'
    today = tm_ut.to_date(tm_ut.time_now(), date_format=date_format)  # date of txn
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    # get today's fund prices
    prices = {n: set_price(t) for n, t in tickers.iteritems()}
    active_funds = tickers.keys()

    df = pd.read_csv(f_name)       # upload html vanguard file on gsheet (negative sign from html busted) and download back

    # check all in data are consistent with funds in cfg
    data_funds = df['Fund'].unique()
    missing = [f for f in data_funds if f not in tickers]  # check the funds in scope are in the data
    if len(missing) > 0:
        print 'ERROR: Missing tickers: ' + str(missing)
        sys.exit(0)

    f_df = df[df['Fund'].isin(active_funds)].copy()  # only funds in scope
    f_df.columns = ['Date', 'Fund', 'Transaction Type', 'Shares', 'Share Price', 'Amount']

    # check input data
    # exchange to, conversion to = buy, exchange from, conversion from = sell
    f_df['Transaction Type'] = f_df['Transaction Type'].apply(lambda x: x if ('Exchange' not in x and 'Conversion' not in x) else ('Buy' if 'To' in x else 'Sell'))
    print 'Initial Transaction Types: ' + str(f_df['Transaction Type'].unique())

    # relabel transaction types
    f_df.replace({'Long-term capital gain': 'LTCG', 'Short-term capital gain': 'STCG', 'Dividend Received': 'Div'}, inplace=True)
    if set(f_df['Transaction Type'].unique()) != {'Buy', 'Sell', 'LTCG', 'STCG', 'Div'}:
        print 'Transaction Types: ' + str(set(f_df['Transaction Type'].unique()))

    # Sales must have negative shares
    # Non-sales: must have positive shares
    # check that this is the case
    # replace broken strings
    f_df['Shares'] = f_df['Shares'].apply(lambda x: locale.atof(x))  # to float
    z = f_df[(f_df['Shares'] <= 0) & (f_df['Transaction Type']).isin(['Buy', 'LTCG', 'STCG', 'Div'])]
    if len(z) > 0:
        print 'ERROR: shares and transaction types'
        print z

    z = f_df[(f_df['Shares'] >= 0) & (f_df['Transaction Type']).isin(['Sell'])]
    if len(z) > 0:
        print 'ERROR: shares and transaction types'
        print z

    z = f_df[f_df['Transaction Type'] == 'Transfer']
    if len(z) > 0:
        print 'Transfer Transactions'
        print z

    # drop empty funds
    sh_df = f_df.groupby('Fund').agg({'Shares': np.sum}).reset_index()
    fund_list = sh_df[np.abs(sh_df['Shares']) > 0.1]['Fund'].values
    f_df = f_df[f_df['Fund'].isin(fund_list)]

    # share price and amount to floats
    f_df['Share Price'] = f_df['Share Price'].apply(lambda x: locale.atof((x.replace('$', ''))))
    f_df['Amount'] = f_df['Amount'].apply(lambda x: locale.atof((x.replace('$', ''))))

    f_df['tax_status'] = f_df.apply(tax_status, today=today, date_format=date_format, axis=1)
    f_df['lot_gain'] = f_df.apply(lot_gain, sale_prices=prices, axis=1)
    f_df['lot_amt'] = f_df.apply(lot_amount, sale_prices=prices, axis=1)
    g_df = f_df.groupby(['Fund', 'tax_status']).apply(fund_smry).reset_index(drop=True)
    g_df.to_csv('/tmp/g_df.csv', index=False)

    p_df = pd.DataFrame({'Fund': map(str, prices.keys()), 'Today Price': prices.values()})
    i_df = g_df.merge(p_df, on='Fund', how='left')
    irr_df = i_df.groupby('Fund').apply(get_irr, t_date=today, d_fmt=date_format, today_vals=None).reset_index(0).reset_index(drop=True)
    z_amt = irr_df['Today$'].apply(lambda x: float(x.replace('$', ''))).sum()  # ttl amt available in funds
    z_irr = get_irr(i_df, today, date_format, today_vals=z_amt)
    z_irr['Fund'] = 'All'
    irr_all = pd.concat([z_irr, irr_df], axis=0)
    irr_all.reset_index(inplace=True, drop=True)
    irr_all.to_csv('/tmp/irr_all.csv', index=False)
    print irr_all.head(len(irr_all))


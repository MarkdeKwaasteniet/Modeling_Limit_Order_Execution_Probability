import pandas as pd
import numpy as np
from datetime import datetime
import h5py
import os


class DataProcessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.df_orders_placed = pd.DataFrame()

    def df_creator(self,
                   filename):  # Create an orderbook dataframe for every month/seperate h5 file.
        data_file = h5py.File(os.path.join(self.folder_path, filename),
                              'r')  # Use h5py package to read in the h5 file.
        self.df_total = pd.DataFrame(data=np.array(data_file[
                                                       '/data/block0_values']))  # Create a
        # DataFrame containing all the values in the orderbook
        self.df_total.columns = np.array(data_file['/data/block0_items'])
        # Fill in the column names using the block0_items: Example:
        # b'bidsP:0|binance-perps|XRP-USDT_PS'
        df_index = pd.DataFrame(data=np.array(data_file['/data/axis1']))
        df_index.columns = ["orderbook_index"]
        self.df_total['timestamp'] = df_index.orderbook_index
        self.df_total.index = pd.to_datetime(self.df_total.timestamp, unit='ns')
        self.df_total = self.df_total.drop(columns=['timestamp'])

        return self.df_total

    def order_placement(self):
        self.df_orders_placed.loc[
            (self.df_orders_placed['sold_orders'] > 0), 'order_subtraction'] = - \
            self.df_orders_placed['sold_orders']

        self.df_orders_placed.loc[(self.df_orders_placed['quantity_change'] < 0) & (
                abs(self.df_orders_placed['quantity_change']) > abs(
            self.df_orders_placed['sold_orders'])), 'order_subtraction'] = self.df_orders_placed[
            'quantity_change']
        self.df_orders_placed['order_subtraction'] = self.df_orders_placed[
            'order_subtraction'].fillna(0)

        self.df_orders_placed.loc[
            (self.df_orders_placed['quantity_change'] > 0), 'order_placed'] = (
                self.df_orders_placed['sold_orders'] + self.df_orders_placed['quantity_change'])
        self.df_orders_placed.loc[((self.df_orders_placed['quantity_change'] < 0) & (
                abs(self.df_orders_placed['quantity_change']) < abs(
            self.df_orders_placed['sold_orders']))), 'order_placed'] = (
                self.df_orders_placed['sold_orders'] + self.df_orders_placed['quantity_change'])

    def threshold_finder(self, placed_order):
        threshold = self.df_orders_placed.loc[str(placed_order.timestamp)].order_quantity
        df_cum_sum = self.df_orders_placed.loc[str(placed_order.timestamp):].iloc[1:, :]

        running_sum = 0
        for future_date in range(1, len(df_cum_sum)):
            running_sum += abs(df_cum_sum.loc[df_cum_sum.index[future_date], 'order_subtraction'])
            try:
                if running_sum >= threshold:
                    if df_cum_sum.loc[df_cum_sum.index[future_date], 'sold_orders'] > 0:
                        return df_cum_sum.loc[df_cum_sum.index[future_date], 'timestamp']
            except Exception as e:
                print(e)
                return np.nan
        return

    def execution_date(self):

        mask = (self.df_orders_placed['order_placed'] > 0)
        self.df_orders_placed.loc[mask, 'timestamp_execution'] = self.df_orders_placed.loc[
            mask].apply(
            lambda placed_order: self.threshold_finder(placed_order), axis=1)

    def calc_execution_time(self, df_trades, df_orderbook):
        df_check = {}
        types_dict = {"bid": -1, "ask": 1}
        columns_list = ['sold_orders', 'order_subtraction', 'order_placed', 'timestamp_execution']

        df_trades_executed = pd.DataFrame(index=df_trades.index)

        for c, item in types_dict.items():
            df_trades_executed[c] = df_trade_volume.loc[df_trade_volume['type'] == item].volume
            self.df_orders_placed = df_orderbook.iloc[:, ((item + 1) * 10 + 10)].diff()
            custom_dates = orderbook_data.index
            custom_sum = df_trades_executed[c].groupby(
                custom_dates[custom_dates.searchsorted(df_trades_executed[c].index)]).sum()
            self.df_orders_placed = self.df_orders_placed.to_frame().join(custom_sum.to_frame())
            self.df_orders_placed = self.df_orders_placed.reset_index()
            self.df_orders_placed['timestamp_shift'] = self.df_orders_placed.timestamp.shift(
                -1).dropna()

            self.df_orders_placed.rename(
                columns={self.df_orders_placed.columns[1]: "quantity_change"},
                inplace=True)
            self.df_orders_placed = self.df_orders_placed.iloc[:(len(df_orderbook) - 1), :]
            for column in columns_list:
                self.df_orders_placed[column] = np.nan

            self.df_orders_placed['sold_orders'] = self.df_orders_placed[c]
            self.df_orders_placed['sold_orders'] = self.df_orders_placed['sold_orders'].fillna(0)
            self.df_orders_placed = self.df_orders_placed.drop(columns=c)
            self.order_placement()

            self.df_orders_placed = self.df_orders_placed.set_index(self.df_orders_placed.timestamp)
            self.df_orders_placed['order_quantity'] = df_orderbook.iloc[:, ((item + 1) * 10 + 10)]

            self.execution_date()

            self.df_orders_placed = self.df_orders_placed.join(
                df_orderbook.iloc[:, ((item + 1) * 10)])
            self.df_orders_placed.rename(columns={self.df_orders_placed.columns[8]: "price"},
                                         inplace=True)
            self.df_orders_placed = self.df_orders_placed.iloc[:, 5:]
            df_check[c] = self.df_orders_placed
        df_merged = df_check['bid'].join(df_check['ask'], lsuffix='_bid', rsuffix='_ask')

        df_merged.to_csv(os.path.join(self.folder_path, "BTC Execution Times"))


FOLDER = r"C:\Users\markkw\Documents\GitHub\Modeling Limit Order Execution Probability\BTC_Data"

ORDERBOOK_FILE = "BTC_ob.h5"
TRADES_FILE = "BTC_trades.h5"

df_processing = DataProcessor(FOLDER)

orderbook_data = df_processing.df_creator(ORDERBOOK_FILE)
trades_data = df_processing.df_creator(TRADES_FILE)

df_trade_volume = trades_data.copy()
columns_dict = {'volume': 1, 'type': 2, 'price': 0}
for c, i in columns_dict.items():
    df_trade_volume[c] = np.nan
    df_trade_volume[c] = df_trade_volume.iloc[:, i].values

df_trade_volume = df_trade_volume.iloc[:, 3:]

df_processing.calc_execution_time(df_trade_volume, orderbook_data)

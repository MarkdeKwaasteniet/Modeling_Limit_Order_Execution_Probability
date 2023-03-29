import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import h5py
import os


class StaticPriceCalculator:
    def __init__(self, df_orderbook: pd.DataFrame()):
        self.df = df_orderbook

    def calc_bid_ask_spread(self):
        spread = self.df['asksP_0'] - self.df['bidsP_0']
        return spread

    def unweighted_mid_price(self):
        price = (self.df['asksP_0'] + self.df['bidsP_0']) / 2
        return price

    def VWAP(self):
        price = ((self.df['bidsP_0'] * self.df['bidsQ_0']) + (
                self.df['asksP_0'] * self.df['asksQ_0'])) / (
                        self.df['asksQ_0'] + self.df['bidsQ_0'])
        return price

    def micro_price(self):
        price = ((self.df['bidsP_0'] * self.df['asksQ_0']) + (
                self.df['asksP_0'] * self.df['bidsQ_0'])) / (
                        self.df['asksQ_0'] + self.df['bidsQ_0'])
        return price

    def calc_imbalance(self):
        # or the other way around ?
        return self.df['asksQ_0'] / self.df['bidsQ_0']

    def deep_price(self, column_P, column_Q, split_point,
                   coins):  # A function to calculate the quantity weighted average mid-point price
        df = self.df.reset_index(drop=True)
        df_coins = {}  # Create dictionary to store the price of every coin separately
        for i in range(0,
                       int(len(coins))):  # Loop over the number of coins in the dataset: in our
            # example this is 6
            df_order_book = df.iloc[:, ((split_point * 2) * i):((split_point * 2) + (
                    split_point * 2) * i)]  # Create a df of 40 columns every loop, which contain
            # bid prices, bid quantities, ask prices, ask quantities

            thresh_Q = list(coins.values())[
                i]  # Set the quantity threshold for a certain coin. Based on its name index in

            # the dictionary "coins"

            # XRP is the first coin, so index is 0. Resulting in a quantity of 4500 XRP -->
            # approx 2000 USD.
            def price_calc(df_split):
                price = []
                for index, row in df_split.iterrows():  # Iterate over the splitted dataframe
                    # which contains only bids or asks (quantity and prices)
                    if df_split.iloc[
                        index, column_Q] >= thresh_Q:  # If quantity is above the determined
                        # threshold, take the first bid/ask price
                        true_price = df_split.iloc[
                                         index, column_P] * thresh_Q  # price will be multiplied
                        # by threshold, at the end of the loop it will be divided by same quantity.

                    else:
                        quantity = thresh_Q  # If quantity is not above to determine threshold,
                        # create while loop in which it will calculate weighted price
                        book_dept = 0  # Book_dept indicates the number of steps the prices has
                        # gone into the orderbook.
                        true_price = 0  # True price is considered the quantity weighted price

                        while quantity > 0:  # While the quantity is not yet equal to the
                            # threshold, go deeper into orderbook
                            if quantity > df_split.iloc[
                                index, book_dept + column_Q]:  # If quantity available is smaller
                                # Then (still) required quantity for threshold:

                                true_price += (df_split.iloc[index, book_dept] * df_split.iloc[
                                    index, book_dept + column_Q])  # Multiply price by the
                                # available quantity
                                quantity -= df_split.iloc[
                                    index, book_dept + column_Q]  # Reduce the required quantity
                                # by the available quantity that is bought

                                if book_dept < 9:  # If the end of the orderbook is not yet met.
                                    book_dept += 1  # Take another step into the orderbook
                                else:
                                    true_price += df_split.iloc[
                                                      index, book_dept] * quantity  # If the end
                                    # of the orderbook is met, use the last quantity/price to
                                    # fill the rest of the required quantity.
                                    quantity -= df_split.iloc[
                                        index, book_dept + column_Q]  # Reduce the required
                                    # quantity by the available quantity that is bought,
                                    # which in this case is all.
                                    break  # Break the fore loop en determine next price

                            else:
                                true_price += df_split.iloc[
                                                  index, book_dept] * quantity  # If the quantity
                                # available is larger than the still required quantity for
                                # threshold, buy rest of quantity needed at this price
                                break  # Break the fore loop en determine next price

                    weighted_price = true_price / thresh_Q  # prices have been multiplied by
                    # threshold quantity, to get the actual price back, divide it by Threshold_Q
                    price.append(weighted_price)  # Append the price to the list of prices
                return price

            bid_price = price_calc(
                df_order_book.iloc[:, :split_point])  # Return the deep prices in the form of a list
            ask_price = price_calc(df_order_book.iloc[:, split_point:])

            # df_coins[list(coins.keys())[i]] =  (np.array(bid_price) + np.array(ask_price)) / 2                          #Calculate the weighted average mid point, by taking the averages of the bid and ask deep prices

        return (np.array(bid_price[:(len(df) - 1)]) + np.array(ask_price[:(len(df) - 1)])) / 2
        # return pd.DataFrame.from_dict(df_coins)


class DataProcessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.df_orders_placed = pd.DataFrame()
        self.df_total = pd.DataFrame()

    def df_creator(self,
                   filename,
                   trades):  # Create an orderbook dataframe for every month/seperate h5 file.
        data_file = h5py.File(os.path.join(self.folder_path, filename),
                              'r')  # Use h5py package to read in the h5 file.
        self.df_total = pd.DataFrame(data=np.array(data_file[
                                                       '/data/block0_values']))  # Create a
        # DataFrame containing all the values in the orderbook
        if trades:
            self.df_total.columns = [str(c).replace("b'", "").replace("'", "") for c in
                                     np.array(data_file['/data/block0_items']).tolist()]
        elif not trades:
            self.df_total.columns = [
                str(c).replace("b'", "").replace("'", "").replace(":", "_").split("|", 1)[0] for c
                in np.array(data_file['/data/block0_items']).tolist()]

        # Fill in the column names using the block0_items: Example:
        # b'bidsP:0|binance-perps|XRP-USDT_PS'
        df_index = pd.DataFrame(data=np.array(data_file['/data/axis1']))
        self.df_total['timestamp'] = df_index.loc[:]
        self.df_total.index = pd.to_datetime(self.df_total.timestamp, unit='ns')
        self.df_total = self.df_total.drop(columns=['timestamp'])

        return self.df_total

    def order_placement(self, c):
        self.df_orders_placed.loc[
            (self.df_orders_placed['sold_orders'] > 0), 'order_subtraction'] = - \
            self.df_orders_placed['sold_orders']

        self.df_orders_placed.loc[(self.df_orders_placed['quantity_change'] < 0) & (
                abs(self.df_orders_placed['quantity_change']) > abs(
            self.df_orders_placed['sold_orders']))
                                  & (self.df_orders_placed["order_price"] == self.df_orders_placed[
            str(c) + "_price_shift"]), 'order_subtraction'] = self.df_orders_placed[
            'quantity_change']
        if c == "bid":
            self.df_orders_placed.loc[(self.df_orders_placed["order_price"] < self.df_orders_placed[
                str(c) + "_price_shift"]), 'order_subtraction'] = -self.df_orders_placed[
                'order_quantity']
        elif c == "ask":
            self.df_orders_placed.loc[(self.df_orders_placed["order_price"] > self.df_orders_placed[
                str(c) + "_price_shift"]), 'order_subtraction'] = -self.df_orders_placed[
                'order_quantity']

        self.df_orders_placed['order_subtraction'] = self.df_orders_placed[
            'order_subtraction'].fillna(0)

        self.df_orders_placed.loc[
            (self.df_orders_placed['quantity_change'] > 0), 'order_placed'] = (
                self.df_orders_placed['sold_orders'] + self.df_orders_placed['quantity_change'])
        self.df_orders_placed.loc[((self.df_orders_placed['quantity_change'] < 0) & (
                abs(self.df_orders_placed['quantity_change']) < abs(
            self.df_orders_placed['sold_orders']))), 'order_placed'] = (
                self.df_orders_placed['sold_orders'] + self.df_orders_placed['quantity_change'])

        # self.df_orders_placed.loc[(self.df_orders_placed['quantity_change'] < 0) & (
        #         abs(self.df_orders_placed['quantity_change']) > abs(
        #     self.df_orders_placed['sold_orders'])), 'order_subtraction'] = self.df_orders_placed[
        #     'quantity_change']
        # self.df_orders_placed['order_subtraction'] = self.df_orders_placed[
        #     'order_subtraction'].fillna(0)
        #
        # self.df_orders_placed.loc[
        #     (self.df_orders_placed['quantity_change'] > 0), 'order_placed'] = (
        #         self.df_orders_placed['sold_orders'] + self.df_orders_placed['quantity_change'])
        # self.df_orders_placed.loc[((self.df_orders_placed['quantity_change'] < 0) & (
        #         abs(self.df_orders_placed['quantity_change']) < abs(
        #     self.df_orders_placed['sold_orders']))), 'order_placed'] = (
        #         self.df_orders_placed['sold_orders'] + self.df_orders_placed['quantity_change'])

    def threshold_finder(self, placed_order, order_size=False):
        if not order_size:
            threshold = self.df_orders_placed.loc[str(placed_order.timestamp)].order_quantity
        else:
            threshold = (self.df_orders_placed.loc[str(placed_order.timestamp)].order_quantity -
                         self.df_orders_placed.loc[
                             str(placed_order.timestamp)].order_placed) + order_size

        df_cumsum = self.df_orders_placed.loc[str(placed_order.timestamp):].iloc[1:, :]

        running_sum = 0
        for future_date in range(1, len(df_cumsum)):
            running_sum += abs(df_cumsum.loc[df_cumsum.index[future_date], 'order_subtraction'])
            try:
                if running_sum >= threshold:
                    if df_cumsum.loc[df_cumsum.index[future_date], 'sold_orders'] > 0:
                        return df_cumsum.loc[df_cumsum.index[future_date], 'timestamp']
            except:
                return np.nan

        # threshold = self.df_orders_placed.loc[str(placed_order.timestamp)].order_quantity
        # df_cum_sum = self.df_orders_placed.loc[str(placed_order.timestamp):].iloc[1:, :]
        #
        # running_sum = 0
        # for future_date in range(1, len(df_cum_sum)):
        #     running_sum += abs(df_cum_sum.loc[df_cum_sum.index[future_date], 'order_subtraction'])
        #     try:
        #         if running_sum >= threshold:
        #             if df_cum_sum.loc[df_cum_sum.index[future_date], 'sold_orders'] > 0:
        #                 return df_cum_sum.loc[df_cum_sum.index[future_date], 'timestamp']
        #     except Exception as e:
        #         print(e)
        #         return np.nan
        # return

    def volume_summation(self, placed_order, column):
        past_minute = placed_order.timestamp - timedelta(seconds=1)
        df_past_volume = self.df_orders_placed.loc[
                         str(past_minute):str(placed_order.timestamp)].iloc[:, :]
        return df_past_volume[column].sum()

    def execution_date(self, size_vector):
        mask = (self.df_orders_placed['order_placed'] > 0)
        for make_size in size_vector[:-1]:
            self.df_orders_placed.loc[mask, ('timestamp_execution_' + str(make_size))] = \
                self.df_orders_placed.loc[mask].apply(
                    lambda placed_order: self.threshold_finder(placed_order,
                                                          float(make_size)), axis=1)

        self.df_orders_placed.loc[mask, ('timestamp_execution_full')] = self.df_orders_placed.loc[
            mask].apply(
            lambda placed_order: self.threshold_finder(placed_order), axis=1)
        self.df_orders_placed.loc[mask, 'past_taking_volume'] = self.df_orders_placed.loc[mask].apply(
            lambda placed_order: self.volume_summation(placed_order, "sold_orders"),
            axis=1)
        self.df_orders_placed.loc[mask, 'past_making_volume'] = self.df_orders_placed.loc[
            mask].apply(
            lambda placed_order: self.volume_summation(placed_order, "order_placed"),
            axis=1)

        # mask = (self.df_orders_placed['order_placed'] > 0)
        # self.df_orders_placed.loc[mask, 'timestamp_execution'] = self.df_orders_placed.loc[
        #     mask].apply(
        #     lambda placed_order: self.threshold_finder(placed_order), axis=1)

    def keep_columns(self, column):
        self.df_orders_placed = self.df_orders_placed.iloc[:, 5:]
        self.df_orders_placed = self.df_orders_placed.drop(columns=[str(column) + "_price_shift"])

    def additional_variables(self, df_orderbook, coins):
        price_calculator = StaticPriceCalculator(df_orderbook)
        self.df_orders_placed['spread'] = price_calculator.calc_bid_ask_spread()
        self.df_orders_placed['deep_price'] = price_calculator.deep_price(0, 10,
                                                                          20, coins)  # .BTC
        self.df_orders_placed['unweighted_price'] = price_calculator.unweighted_mid_price()
        self.df_orders_placed['micro_price'] = price_calculator.micro_price()
        self.df_orders_placed['volume_weighted_price'] = price_calculator.VWAP()
        self.df_orders_placed['inventory_bid'] = df_orderbook.iloc[:, 10:20].sum(axis=1)
        self.df_orders_placed['inventory_ask'] = df_orderbook.iloc[:, 30:40].sum(axis=1)
        self.df_orders_placed['imbalance'] = price_calculator.calc_imbalance()
        self.df_orders_placed = self.df_orders_placed.reset_index()

    def calc_execution_time(self, df_trades, df_orderbook, coins, execution_sec, size_vector):
        df_check = {}

        types_dict = {"bid": -1, "ask": 1}
        columns_list = ['sold_orders', 'order_subtraction', 'order_placed']

        df_trades_executed = pd.DataFrame(index=df_trades.index)

        for c, item in types_dict.items():
            df_trades_executed[c] = df_trades.loc[df_trades['buy'] == item].volume

            self.df_orders_placed = df_orderbook[str(c) + "sQ_0"].diff().rename("quantity_change")

            take_orders_sorted = df_trades_executed[c].groupby(df_orderbook.index[
                                                                   df_orderbook.index.searchsorted(
                                                                       df_trades_executed[
                                                                           c].index)]).sum()
            self.df_orders_placed = self.df_orders_placed.to_frame().join(
                take_orders_sorted.to_frame())

            self.df_orders_placed = self.df_orders_placed.reset_index()
            self.df_orders_placed['timestamp_shift'] = self.df_orders_placed.timestamp.shift(
                -1).dropna()
            self.df_orders_placed = self.df_orders_placed.iloc[:(len(df_orderbook) - 1), :]

            self.df_orders_placed[columns_list] = np.nan

            self.df_orders_placed['sold_orders'] = self.df_orders_placed[c].fillna(0)
            self.df_orders_placed = self.df_orders_placed.drop(columns=c)

            self.df_orders_placed = self.df_orders_placed.set_index(self.df_orders_placed.timestamp)

            self.df_orders_placed['order_price'] = df_orderbook[str(c) + "sP_0"]
            self.df_orders_placed[str(c) + "_price_shift"] = self.df_orders_placed[
                self.df_orders_placed.columns[-1]].shift(-1)
            self.df_orders_placed['order_quantity'] = df_orderbook[str(c) + "sQ_0"]

            self.order_placement(c)
            self.execution_date(size_vector)
            self.df_orders_placed['type_order'] = item

            self.keep_columns(c)

            self.additional_variables(df_orderbook, coins)
            df_check[c] = self.df_orders_placed[self.df_orders_placed['order_placed'].notna()]

        df_merged = pd.concat([df_check['bid'], df_check['ask']], ignore_index=True)
        for make_size in size_vector:
            df_merged["timestamp_execution_" + str(make_size)] = pd.to_datetime(
                df_merged["timestamp_execution_" + str(make_size)])
            df_merged["execution_" + str(make_size)] = 0
            df_merged.loc[(df_merged["timestamp_execution_" + str(make_size)] - df_merged['timestamp'] < timedelta(seconds=execution_sec)), "execution_" + str(
                make_size)] = 1
            df_merged = df_merged.drop(columns=["timestamp_execution_" + str(make_size)])

        df_merged.drop(columns=['timestamp']).to_csv(
            os.path.join(self.folder_path, "BTC Execution Times.csv"))

        # df_check = {}
        # types_dict = {"bid": -1, "ask": 1}
        # columns_list = ['sold_orders', 'order_subtraction', 'order_placed', 'timestamp_execution']
        #
        # df_trades_executed = pd.DataFrame(index=df_trades.index)
        #
        # for c, item in types_dict.items():
        #     df_trades_executed[c] = df_trade_volume.loc[df_trade_volume['type'] == item].volume
        #     self.df_orders_placed = df_orderbook.iloc[:, ((item + 1) * 10 + 10)].diff()
        #     custom_dates = orderbook_data.index
        #     custom_sum = df_trades_executed[c].groupby(
        #         custom_dates[custom_dates.searchsorted(df_trades_executed[c].index)]).sum()
        #     self.df_orders_placed = self.df_orders_placed.to_frame().join(custom_sum.to_frame())
        #     self.df_orders_placed = self.df_orders_placed.reset_index()
        #     self.df_orders_placed['timestamp_shift'] = self.df_orders_placed.timestamp.shift(
        #         -1).dropna()
        #
        #     self.df_orders_placed.rename(
        #         columns={self.df_orders_placed.columns[1]: "quantity_change"},
        #         inplace=True)
        #     self.df_orders_placed = self.df_orders_placed.iloc[:(len(df_orderbook) - 1), :]
        #     for column in columns_list:
        #         self.df_orders_placed[column] = np.nan
        #
        #     self.df_orders_placed['sold_orders'] = self.df_orders_placed[c]
        #     self.df_orders_placed['sold_orders'] = self.df_orders_placed['sold_orders'].fillna(0)
        #     self.df_orders_placed = self.df_orders_placed.drop(columns=c)
        #     self.order_placement(c)
        #
        #     self.df_orders_placed = self.df_orders_placed.set_index(self.df_orders_placed.timestamp)
        #     self.df_orders_placed['order_quantity'] = df_orderbook.iloc[:, ((item + 1) * 10 + 10)]
        #
        #     self.execution_date()
        #
        #     self.df_orders_placed = self.df_orders_placed.join(
        #         df_orderbook.iloc[:, ((item + 1) * 10)])
        #     self.df_orders_placed.rename(columns={self.df_orders_placed.columns[8]: "price"},
        #                                  inplace=True)
        #     self.df_orders_placed = self.df_orders_placed.iloc[:, 5:]
        #     df_check[c] = self.df_orders_placed
        # df_merged = df_check['bid'].join(df_check['ask'], lsuffix='_bid', rsuffix='_ask')
        #
        # df_merged.to_csv(os.path.join(self.folder_path, "BTC Execution Times.csv"))


FOLDER = r"C:\Users\markkw\Documents\GitHub\Modeling Limit Order Execution Probability\BTC_Data"

ORDERBOOK_FILE = "BTC_ob.h5"
TRADES_FILE = "BTC_trades.h5"

COINS = {'BTC': 0.1}
EXECUTION_SEC = 2
SIZE_VECT = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, "full"]

df_processing = DataProcessor(FOLDER)

orderbook_data = df_processing.df_creator(ORDERBOOK_FILE, False)
trades_data = df_processing.df_creator(TRADES_FILE, True)

df_processing.calc_execution_time(trades_data, orderbook_data, COINS, EXECUTION_SEC, SIZE_VECT)

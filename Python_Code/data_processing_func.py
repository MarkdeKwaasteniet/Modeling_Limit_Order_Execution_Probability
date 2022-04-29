import pandas as pd
import numpy as np
import h5py

def df_creator(file_path):                                                                   #Create an orderbook dataframe for every month/seperate h5 file.
    data_file = h5py.File(file_path,'r')                                                     #Use h5py package to read in the h5 file.
    df_total = pd.DataFrame(data = np.array(data_file['/data/block0_values']))          #Create a DataFrame containing all the values in the orderbook
    df_total.columns = np.array(data_file['/data/block0_items'])   
                         #Fill in the column names using the block0_items: Example: b'bidsP:0|binance-perps|XRP-USDT_PS'
    df_index= pd.DataFrame(data = np.array(data_file['/data/axis1']))       
    df_index.columns = ["orderbook_index"]
    df_total['timestamp'] = df_index.orderbook_index
    df_total.index = pd.to_datetime(df_total.timestamp,unit='ns')
    df_total = df_total.drop(columns=['timestamp'])
    return df_total    

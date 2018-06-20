import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Global variables
folder_name = "/data"
start_date = "2014-01-01"
end_date = "2018-02-23"


def get_data():
    """
    Takes no arguments.
    Returns data frame with preprocessed inputs, and target output.
    """
    input_dfs = []

    # get data folder
    parent_path = get_parent_dir(os.path.abspath(__file__), 2)
    data_folder = str(parent_path + os.path.join(folder_name))
    assert os.path.exists(data_folder) == True, "The specified data folder cannot be located."

    # Load and pre process input data
    # Descriptive statistics used (per http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155133):
    #     * BIAS(n): Measures the divergence of the current log return from an n-day moving 
    #                 average of log returns. We let n = 6. 
    #     * PSY(n): Psychological line is a proxy for market sentiment.
    #     * ASY(n): The average return in the last n days.
    #     * OBV: The average return in the last n days.

    # ASX SPI 200 Index Futures, Continuous Contract #2 (AP2)
    df_asx_200 = pd.read_csv(os.path.join(data_folder, "AXJO.csv"), index_col=0, parse_dates=True)
    df_asx_200 = process_input_df(df_asx_200, 'AXJO', 'Close')

    df_asx_200['log_r'] = np.log(df_asx_200['close']).diff() * 100
    df_asx_200['ma5_r'] = df_asx_200['log_r'].shift(1).rolling(window=5).sum()
    df_asx_200['bias6_r'] = df_asx_200['log_r'].shift(1) - df_asx_200['log_r'].shift(1).rolling(window=6).sum()
    df_asx_200['psy12_r'] = (df_asx_200['log_r'].shift(1).rolling(window=12, min_periods=12)
                             .agg(lambda x: (x > 0).sum())) / 12

    # ASX Trading Volumes (Daily)
    df_asx_vol = pd.read_csv(os.path.join(data_folder, "asx_volume.csv"), index_col=0, parse_dates=True)
    input_dfs.append(process_input_df(df_asx_vol, 'volume', 'AS51 Index'))

    # Dow Jones Industrial Average
    df_dji = pd.read_csv(os.path.join(data_folder, "DJI.csv"), index_col=0, parse_dates=True)
    input_dfs.append(process_input_df(df_dji, 'dji', 'Close'))

    # Create final data frame with all required input and output data
    output_fml = np.where(df_asx_200['log_r'] > 0, 1, 0)
    df_asx_200.insert(loc=1, column='output', value=output_fml)
    df_asx_200.drop('close', axis=1, inplace=True)
    df_asx_200.drop('log_r', axis=1, inplace=True)

    for df in input_dfs:
        df.drop('close', axis=1, inplace=True)
        df_asx_200 = pd.merge(df_asx_200, df, how='inner', left_index=True, right_index=True)

    # Drop all rows with na values - will not effect drivers for date
    df_asx_200 = df_asx_200.dropna(axis=0, how='any')

    return df_asx_200


def get_parent_dir(file_path, ancestor):
    if ancestor == 0:
        return file_path
    else:
        return get_parent_dir(os.path.abspath(os.path.join(file_path, os.pardir)), ancestor - 1)


def drop_columns(df, exclusion):
    """
    Takes data frame, and list of column ids.
    Drops all columns not specified in exclusion.
    Returns remaining data frame.
    """
    for c in df.columns.values:
        if c not in exclusion:
            df.drop(c, axis=1, inplace=True)
    return df


def process_input_df(df_raw, market, price_col_id):
    """
    Takes data frame, market name and last price column id.
    Changes closing price to a uniform id.
    Calculates daily log return, 5/25 day moving average.
    """
    # drop all non required columns from data frame
    df = drop_columns(df_raw, [price_col_id])

    # rename last price column to uniform convention
    df = df.rename(columns={price_col_id: 'close'})

    # calculate required indicators
    df[market + '_rt1'] = np.log(df['close']).diff().shift(1) * 100
    df[market + '_asy5_r'] = df[market + '_rt1'].rolling(window=5).sum() / 5
    df[market + '_asy4_r'] = df[market + '_rt1'].rolling(window=4).sum() / 4
    df[market + '_asy3_r'] = df[market + '_rt1'].rolling(window=3).sum() / 3
    df[market + '_asy2_r'] = df[market + '_rt1'].rolling(window=2).sum() / 2

    return df


def split_data(df, train_prop):
    """
    Takes data frame, and returns training and test dataframes based on specified proportion.
    """
    # Create random Tensors to hold inputs and outputs, and wrap them in Variables
    train_df = df.sample(frac=train_prop)
    test_df = df.loc[~df.index.isin(train_df.index)]
    return train_df, test_df


def plot_data(test_results=None, training_results=None):
    # plot test results
    if test_results:
        lists = sorted(test_results.items(), reverse=True)  # sorted by key, return a list of tuples
        x, y = zip(*lists)  # unpack a list of pairs into two tuples

        plt.plot(x, y)
        plt.xlim(max(x), min(x))
        plt.ylabel('Testing Accuracy')
        plt.xlabel('Number of Input Neurons')
        plt.title("Accuracy vs Input Neurons")
        plt.show()

    # plot training result
    if training_results:
        x1 = []
        y1 = []
        for k in training_results.keys():
            x1.append(k)
            y1.append(training_results[k].data)

        plt.plot(x1, y1)
        plt.xlim(max(x), min(x))
        plt.ylabel('Mean Square Error')
        plt.xlabel('Number of Input Neurons')
        plt.title("Error vs Input Neurons")
        plt.show()

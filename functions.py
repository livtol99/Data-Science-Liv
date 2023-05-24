
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



def create_splits(df):
    first_test_date = pd.Timestamp('2005-01-01')
    train_clus2, test_clus2 = [], []

    # Group the data by 'Country' column
    grouped = df.groupby('Country')

    # Iterate over each group
    for _, group_data in grouped:
        # Sort the group data by the 'Year' column
        group_data = group_data.sort_values('Year')

        # Split the data based on the first_test_date
        train_data = group_data[group_data['Year'] < first_test_date]
        test_data = group_data[group_data['Year'] >= first_test_date]

        # Append the train and test data to the respective sets
        train_clus2.append(train_data)
        test_clus2.append(test_data)

    # Concatenate the train and test data sets into single DataFrames
    train_df = pd.concat(train_clus2)
    test_df = pd.concat(test_clus2)
    
    return train_df, test_df


def yr_aggregate_train(train_df):
    # Perform yearly aggregation across all individual time series
    agg_df_train = train_df.groupby('Year')['LE'].median().reset_index()
    agg_df_train.rename({'Year':'ds', 'LE':'y'}, axis=1, inplace=True)
    
    return agg_df_train

def yr_aggregate_test(test_df):
    # Perform yearly aggregation across all individual time series
    agg_df_test = test_df.groupby('Year')['LE'].median().reset_index()
    agg_df_test.rename({'Year':'ds', 'LE':'y'}, axis=1,inplace=True)
    return agg_df_test

def rename_merge_concat(train_df, test_df, agg_df_train, agg_df_test, predictions):
    # Rename columns to fix the issues
    agg_df_train.rename({'y':'y_agg_tr'}, axis=1, inplace=True)
    agg_df_test.rename({'y':'y_agg_tr'}, axis=1, inplace=True)
    train_df.rename({'Year':'ds','LE':'y_tr'}, axis=1, inplace=True)
    test_df.rename({'Year':'ds', 'LE':'y_tst'}, axis=1, inplace=True)

    # Merge train_df and aggregated train df
    merged_train_df = pd.merge(train_df, agg_df_train, on='ds', how='inner')

    # Compute adjustment series for the merged test and train data sets
    merged_train_df['delta_val'] = merged_train_df['y_tr'] - merged_train_df['y_agg_tr']

    # Merge the "yhat" column from "predictions" with "test_df"
    merged_test_df = pd.merge(test_df, predictions[['ds', 'yhat']], on='ds', how='inner')

    # Concatenate merged_train_df and merged_test_df
    df_actual_all = pd.concat([merged_train_df, merged_test_df], ignore_index=True)

    return merged_train_df, merged_test_df, df_actual_all

def get_linear_model_pred(train_test_df, test_dat):
    # Sort the train_test_df based on the date column
    train_test_df = train_test_df.sort_values(['ds']).reset_index(drop=True)

    # Create a column 't' with values from 0 to the number of rows in train_test_df
    train_test_df['t'] = np.arange(0, train_test_df.shape[0])

    # Create a column 't2' which is the square of the 't' values
    train_test_df['t2'] = train_test_df['t'] ** 2

    # Extract the training data (excluding test period) from train_test_df
    train_df = train_test_df.loc[~train_test_df['ds'].astype(str).isin(test_dat['ds'].astype(str).values.tolist())].reset_index(drop=True)

    # Extract the test data from train_test_df
    test_df = train_test_df.loc[train_test_df['ds'].astype(str).isin(test_dat['ds'].astype(str).values.tolist())].reset_index(drop=True)

    # Create a LinearRegression object and fit the model using the training data
    lm_obj = LinearRegression().fit(train_df[['t', 't2']], train_df['delta_val'])

    # Make predictions for the test data
    test_delta_pred = lm_obj.predict(test_df[['t', 't2']])

    # Create a DataFrame with the predicted values and corresponding months in the test period
    return pd.DataFrame({'month': test_df['ds'].values, 'test_delta_pred': test_delta_pred})

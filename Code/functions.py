
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



def create_splits(df):
    first_test_date = pd.Timestamp('2012-01-01')
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


def create_representative_train(train_df):
    # Perform yearly aggregation across all individual time series
    representative_df_train = train_df.groupby('Year')['LE'].median().reset_index()
    representative_df_train.rename({'Year':'ds', 'LE':'y'}, axis=1, inplace=True)
    
    return representative_df_train

def create_representative_test(test_df):
    # Perform yearly aggregation across all individual time series
    representative_df_test = test_df.groupby('Year')['LE'].median().reset_index()
    representative_df_test.rename({'Year':'ds', 'LE':'y'}, axis=1,inplace=True)
    return representative_df_test

def create_adjustment_series(train_df, test_df, representative_df_train, representative_df_test, representative_forecasts):
    # Rename columns to avoid issues during merge
    representative_df_train.rename({'y':'y_repr_tr'}, axis=1, inplace=True)
    representative_df_test.rename({'y':'y_repr_tst'}, axis=1, inplace=True)
    train_df.rename({'Year':'ds','LE':'y_tr'}, axis=1, inplace=True)
    test_df.rename({'Year':'ds', 'LE':'y_tst'}, axis=1, inplace=True)

    # Merge train_df and representative train df
    merged_train_df = pd.merge(train_df, representative_df_train, on='ds', how='inner')

    # Compute adjustment series for the train data set by subtracting the representative ts from the actual ts
    merged_train_df['adjusted_val'] = merged_train_df['y_tr'] - merged_train_df['y_repr_tr']

    # Merge the "yhat" (representative forecasts) column  with the est_df
    merged_test_df = pd.merge(test_df, representative_forecasts[['ds', 'yhat']], on='ds', how='inner')

    # Concatenate merged_train_df and merged_test_df
    df_actual_all = pd.concat([merged_train_df, merged_test_df], ignore_index=True)

    return df_actual_all

def get_linear_model_pred(df_actual_all, test_dat):
    # Sort the df_actual_all based on the date column
    df_actual_all = df_actual_all.sort_values(['ds']).reset_index(drop=True)

    # Create a column 't' with values from 0 to the number of rows in df_actual_all
    df_actual_all['t'] = np.arange(0, df_actual_all.shape[0])

    # Create a column 't2' which is the square of the 't' values
    df_actual_all['t2'] = df_actual_all['t'] ** 2

    # Extract the training data (excluding test period) from df_actual_all
    train_df = df_actual_all.loc[~df_actual_all['ds'].astype(str).isin(test_dat['ds'].astype(str).values.tolist())].reset_index(drop=True)

    # Extract the test data from df_actual_all
    test_df = df_actual_all.loc[df_actual_all['ds'].astype(str).isin(test_dat['ds'].astype(str).values.tolist())].reset_index(drop=True)

    # Create a LinearRegression object and fit the model using the training data
    lm_obj = LinearRegression().fit(train_df[['t', 't2']], train_df['adjusted_val'])

    # Make predictions for the test data
    adjustment_Forecasts_test = lm_obj.predict(test_df[['t', 't2']])

    # Create a DataFrame with the predicted values and corresponding months in the test period
    return pd.DataFrame({'Year': test_df['ds'].values, 'adjustment_Forecasts_test': adjustment_Forecasts_test})

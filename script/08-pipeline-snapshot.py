import pandas as pd
import numpy as np
import os
import datetime
from dateutil.relativedelta import relativedelta
import warnings


def setup_pandas_options():
    """Configure pandas options and warnings"""
    pd.set_option('mode.chained_assignment', None)
    pd.set_option('future.no_silent_downcasting', True)
    warnings.simplefilter(action='ignore', category=FutureWarning)


def load_and_preprocess_data(file_path):
    """Load and preprocess the data"""
    # Read the data
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    # Preprocess column names and types
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    return df

# Function to get the first days of each month between two dates
def get_first_days_of_months(start_date, end_date):
    # List to store the first days of each month
    first_days = []
    
    # Start from the first day of the start date's month
    current_date = start_date.replace(day=1)
    
    # Loop until the current_date is past the end_date
    while current_date <= end_date:
        first_days.append(current_date)
        current_date += relativedelta(months=1)  # Add one month
        
    return first_days

def snapshot_past_month_begin(df: pd.DataFrame, time_snapshot: datetime.datetime, time_shot: int = 1) -> pd.DataFrame:

    # Get list of unique customers until the time_snapshot
    customers_until_time_snapshot = df[df['InvoiceDate'] < time_snapshot]['CustomerID'].dropna().unique()

    # Filter the data to get the snapshot of one month
    df_filter = df[(df['InvoiceDate'] > time_snapshot - relativedelta(months=time_shot)) & (df['InvoiceDate'] <= time_snapshot- relativedelta(months=time_shot-1))]

    # Create a new column to store the total amount of each transaction
    df_filter['total_amount'] = df_filter['Quantity'] * df_filter['UnitPrice']  

    # Create a new Dataframe to store the snapshot of the data  
    df_snapshot = pd.DataFrame(columns=['customer_id', 'total_successful_amount_past_1_month', 'num_successful_orders_past_1_month'])

    # List of all customers in the filtered data
    lst_customer_filter = df_filter['CustomerID'].unique()

    # Get all the customer before the time snapshot-time_shot
    customers_until_past_timeshot = df[(df['InvoiceDate'] < time_snapshot - relativedelta(months=time_shot))]["CustomerID"].unique()
    
  
    # Customers who have successful orders during the time_snapshot-time_shot
    df_snapshot = pd.DataFrame(columns=['customer_id', f'total_successful_amount_past_{time_shot}_month', f'num_successful_orders_past_{time_shot}_month'])
    df_filter_agg = df_filter.groupby('CustomerID').aggregate({'total_amount': 'sum', 'InvoiceDate': 'count'}).reset_index()
    df_filter_agg.rename(columns={'CustomerID':'customer_id', 'InvoiceDate': f'num_successful_orders_past_{time_shot}_month', 'total_amount': f'total_successful_amount_past_{time_shot}_month'}, inplace=True)
    df_snapshot = pd.concat([df_snapshot, df_filter_agg], ignore_index=True) if df_snapshot.shape[0] > 0 else df_filter_agg


    # Customers who have successful orders in the past of the time_snapshot-time_shot but have no successful orders in the filtered data
    df_successful_order = pd.DataFrame(columns=['customer_id', f'total_successful_amount_past_{time_shot}_month', f'num_successful_orders_past_{time_shot}_month'])
    lst_no_orders = [customer_id for customer_id in customers_until_time_snapshot if (customer_id not in lst_customer_filter) & (customer_id in customers_until_past_timeshot)]    
    df_successful_order['customer_id'] = lst_no_orders
    df_successful_order.fillna(0, inplace=True)

    # Customers who have no successful orders in the past of the time_snapshot-time_shot
    df_no_information = pd.DataFrame(columns=['customer_id', f'total_successful_amount_past_{time_shot}_month', f'num_successful_orders_past_{time_shot}_month'])
    lst_no_information = [customer_id for customer_id in customers_until_time_snapshot if (customer_id not in lst_customer_filter) & (customer_id not in customers_until_past_timeshot)]
    df_no_information['customer_id'] = lst_no_information
    

    df_snapshot = pd.concat([df_snapshot, df_successful_order, df_no_information], ignore_index=True) if df_snapshot.shape[0] > 0 else pd.concat([df_successful_order, df_no_information], ignore_index=True)

    # Add the time_snapshot to the snapshot dataframe
    df_snapshot['time_snapshot'] = time_snapshot    

    return df_snapshot

def snapshot_future_month_begin(df: pd.DataFrame, time_snapshot: datetime.datetime, time_shot: int = 1) -> pd.DataFrame:

    # Get list of unique customers until the time_snapshot
    customers_until_time_snapshot = df[df['InvoiceDate'] < time_snapshot]['CustomerID'].dropna().unique()    

    # Filter the data to get the snapshot of one month
    df_filter = df[(df['InvoiceDate'] > time_snapshot + relativedelta(months=time_shot-1)) & (df['InvoiceDate'] <= time_snapshot + relativedelta(months=time_shot))]
    df_filter = df_filter[df_filter.CustomerID.isin(customers_until_time_snapshot)]
    
    # Maximal date of the data
    max_date = df['InvoiceDate'].max()
    # Check if snapshot time is in the time frame of data or not
    if time_snapshot + relativedelta(months=time_shot-1) < max_date:

        # Create a new column to store the total amount of each transaction
        df_filter['total_amount'] = df_filter['Quantity'] * df_filter['UnitPrice']  

        # Create a new Dataframe to store the snapshot of the data  
        df_snapshot = pd.DataFrame(columns=['customer_id', f'total_successful_amount_future_{time_shot}_month', f'num_successful_orders_future_{time_shot}_month'])

        # List of all customers in the filtered data
        lst_customer_filter = df_filter['CustomerID'].unique()

    
        df_snapshot = pd.DataFrame(columns=['customer_id', f'total_successful_amount_future_{time_shot}_month', f'num_successful_orders_future_{time_shot}_month'])
        df_filter_agg = df_filter.groupby('CustomerID').aggregate({'total_amount': 'sum', 'InvoiceDate': 'count'}).reset_index()
        df_filter_agg.rename(columns={'CustomerID':'customer_id', 'InvoiceDate': f'num_successful_orders_future_{time_shot}_month', 'total_amount': f'total_successful_amount_future_{time_shot}_month'}, inplace=True)
        df_snapshot = pd.concat([df_snapshot, df_filter_agg], ignore_index=True) if df_snapshot.shape[0] > 0 else df_filter_agg
        
        # Customers who have  have no successful orders in the filtered data
        df_successful_order = pd.DataFrame(columns=['customer_id', f'total_successful_amount_future_{time_shot}_month', f'num_successful_orders_future_{time_shot}_month'])
        lst_no_orders = [customer_id for customer_id in customers_until_time_snapshot if (customer_id not in lst_customer_filter)]    
        df_successful_order['customer_id'] = lst_no_orders
        df_successful_order.fillna(0, inplace=True)

        df_snapshot = pd.concat([df_snapshot, df_successful_order], ignore_index=True) if df_snapshot.shape[0] > 0 else df_successful_order

        # Add the time_snapshot to the snapshot dataframe
            
    else: # Fill data with NaN
        df_snapshot = pd.DataFrame(columns=['customer_id', f'total_successful_amount_future_{time_shot}_month', f'num_successful_orders_future_{time_shot}_month', 'time_snapshot'])
        df_snapshot['customer_id'] = customers_until_time_snapshot 

    df_snapshot['time_snapshot'] = time_snapshot
    return df_snapshot

# Function to that have snapshot_time as input and return the snapshot of the data from the past and the future
## Input
##      df: the original data
##      snapshot_time: the time of the snapshot
##      past_time_shot: the number of time_spots of the snapshot in the past (1 month ago, 2 months ago, 3 months ago, etc.)
##      future_time_shot: the number of time_spots of the snapshot in the future (1 month later, 2 months later, 3 months later, etc.)
## Output
##      df_snapshot: the snapshot of the data at the given time in the past and the future. The duration of the snapshot is one month
##                   The snapshot includes the total amount of successful orders and the number of successful orders for each customer


def snap_shot_past_future(df: pd.DataFrame, snapshot_time: datetime.datetime, past_time_shot: int = 2, future_time_shot: int = 2) -> pd.DataFrame:
    
     # Get the snapshot of the data in the past at the snapshot_time
    df_past= snapshot_past_month_begin(df, snapshot_time, time_shot=1) # Get the snapshot one month ago

    # Loop to get the snapshots of the data in the past at the snapshot_time
    for i in range(2, past_time_shot+1):
        df_past_temp = snapshot_past_month_begin(df, snapshot_time, time_shot=i) # Get the snapshot i months ago
        df_past_temp.drop(columns=['time_snapshot'], inplace=True) # Drop the time_snapshot column
        #df_past_temp.rename(columns={'total_successful_amount_past_1_month': f'total_successful_amount_past_{i}_month', 'num_successful_orders_past_1_month': f'num_successful_orders_past_{i}_month'}, inplace=True) # Rename the columns name
        df_past = pd.merge(df_past_temp, df_past,  on='customer_id', how='right') # Merge the past snapshots on the customer_id

       

    # Loop to get the snapshots of the data in the future at the snapshot_time
    for i in range(1, future_time_shot+1):
        df_future_temp = snapshot_future_month_begin(df, snapshot_time,  time_shot=i) # Get the snapshot i months later
        df_future_temp.drop(columns=['time_snapshot'], inplace=True) # Drop the time_snapshot column   
        df_past = pd.merge(df_past, df_future_temp,  on='customer_id', how='left') # Merge the future snapshots on the customer_id

    
    # Reorder the columns snapshot_time to the end of the dataframe
    col = df_past.pop('time_snapshot')
    df_past['time_snapshot'] = col
        
    return df_past


def RFM_feature(df: pd.DataFrame, snapshot_time: datetime.datetime) -> pd.DataFrame:
    # Create features from the RFM model: Recency, Frequency, and Monetary
    # Get the snapshot of the data in the past at the snapshot_time

    # Get list of unique customers until the time_snapshot
    
    df["total_amount"] = df["Quantity"] * df["UnitPrice"]
    df_until_time_snapshot = df[df['InvoiceDate'] < snapshot_time]    
    df_RFM = df_until_time_snapshot.groupby('CustomerID').agg(
        recency=('InvoiceDate', lambda x: (snapshot_time - x.max()).days),
        frequency=('Quantity', 'count'),
        monetary=('total_amount', 'sum')    
        ).reset_index()
    df_RFM["RecencyScore"] = pd.qcut(df_RFM['recency'].rank(method="first"),5, labels=[5,4,3,2,1])

    df_RFM["FrequencyScore"] = pd.qcut(df_RFM['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    df_RFM["MonetaryScore"] = pd.qcut(df_RFM['monetary'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    df_RFM.rename(columns={'CustomerID':'customer_id'}, inplace=True)
    
    return df_RFM

def snap_shot_past_future_RFM(df: pd.DataFrame, snapshot_time: datetime.datetime, past_time_shot: int = 2, future_time_shot: int = 2) -> pd.DataFrame:
    
    # Create snapshot and RFM features
    df_past_future = snap_shot_past_future(df, snapshot_time, past_time_shot, future_time_shot)
    df_RFM = RFM_feature(df, snapshot_time)

    # Merge the snapshot of the data and the RFM features
    df_past_future_RFM = pd.merge(df_past_future, df_RFM, on='customer_id', how='left')
    
    return df_past_future_RFM


# Function to create the feature mart

def create_feature_mart(df, past_time_shot=3, future_time_shot=2):
    # Get time snapshots
    
    time_snapshots = df['InvoiceDate'].unique()
    time_min = time_snapshots.min().date()
    time_max = time_snapshots.max().date()

    # Create the beginning and ending time of the snapshots
    time_begin = pd.to_datetime(time_min.replace(day=1) + relativedelta(months=1))
    time_end = pd.to_datetime(time_max.replace(day=1)+ relativedelta(months=1))

    # Get the first days of each month between the beginning and ending time 
    first_days = get_first_days_of_months(time_begin, time_end)

    # Create a new Dataframe to store the snapshot of the data  
    df_snapshot_all = pd.DataFrame(columns=['customer_id', 'total_successful_amount_past_1_month', 'num_successful_orders_past_1_month', 'time_snapshot'])

    # Get the snapshot of the data at each time in the past
    for time_snapshot in first_days:
        df_snapshot = snap_shot_past_future_RFM(df, time_snapshot, past_time_shot=past_time_shot, future_time_shot=future_time_shot)        
        df_snapshot_all = pd.concat([df_snapshot_all, df_snapshot], ignore_index=True) if df_snapshot_all.shape[0] > 0 else df_snapshot
    return df_snapshot_all

def main():
    # Setup
    setup_pandas_options()
    
    # Load and preprocess data
    # input_file = 'd:/code/data/data.csv'
    
    
    # Data with two years of data
    input_file = 'd:/code/data/online_retail_II.csv'
    
    
    output_file = 'd:/code/data/customer_behavior_ecom_snapshot_FRM_two_years.csv'
    
    df = load_and_preprocess_data(input_file)
    
    # Create feature mart
    df_feature_mart = create_feature_mart(df, past_time_shot=5, future_time_shot=2)
    
    
    # Save results
    df_feature_mart.to_csv(output_file, index=False)
    print(f"Feature mart saved to {output_file}")
   


if __name__ == "__main__":
    main()
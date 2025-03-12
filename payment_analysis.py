import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

def load_data(db_path='payment_data.db'):
    """Load transaction data from SQLite database"""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM transactions"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Fix the timestamp - SQLite doesn't store proper datetimes
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def analyze_conversion_funnel(df):
    """Analyze the conversion funnel to identify drop-off points"""
    
    # Count how many unique transactions make it to each stage
    funnel_data = df.groupby('funnel_stage')['transaction_id'].nunique().reset_index()
    
    # We expect these stages in this order
    funnel_stages = ['initiated', 'details_entered', 'otp_verification', 'processing', 'completed']
    funnel_counts = []
    for stage in funnel_stages:
        count = funnel_data[funnel_data['funnel_stage'] == stage]['transaction_id'].values
        funnel_counts.append(count[0] if len(count) > 0 else 0)
    
    # Calculate where users are falling off
    initial_count = funnel_counts[0]
    drop_off_rates = []
    for i in range(len(funnel_counts) - 1):
        if funnel_counts[i] > 0:
            drop_rate = (funnel_counts[i] - funnel_counts[i+1]) / funnel_counts[i] * 100
        else:
            drop_rate = 0
        drop_off_rates.append(drop_rate)
    
    return {
        'stages': funnel_stages,
        'counts': funnel_counts,
        'drop_off_rates': drop_off_rates
    }

def detect_anomalies(df, method='zscore', threshold=3.0):
    """Detect anomalies in transaction data
    
    Parameters:
    -----------
    df : DataFrame
        Transaction data
    method : str
        Method to use for anomaly detection ('zscore', 'iqr')
    threshold : float
        Threshold for z-score method or multiplier for IQR method
    
    Returns:
    --------
    DataFrame with anomaly flag
    """
    # We only care about completed transactions for anomaly detection
    completed_tx = df[df['funnel_stage'] == 'completed'].copy()
    
    # Group by hour to spot weird patterns in volume
    completed_tx['hour'] = completed_tx['timestamp'].dt.floor('H')
    hourly_counts = completed_tx.groupby('hour')['transaction_id'].count()
    
    # Choose the anomaly detection method
    if method == 'zscore':
        # Z-score finds values far from the mean
        z_scores = np.abs(stats.zscore(hourly_counts))
        hourly_anomalies = hourly_counts[z_scores > threshold].index.tolist()
    else:
        # IQR finds values outside the typical quartile range
        Q1 = hourly_counts.quantile(0.25)
        Q3 = hourly_counts.quantile(0.75)
        IQR = Q3 - Q1
        hourly_anomalies = hourly_counts[(hourly_counts < (Q1 - threshold * IQR)) | 
                                        (hourly_counts > (Q3 + threshold * IQR))].index.tolist()
    
    # Flag transactions that happened during anomalous hours
    df['hourly_anomaly'] = df['timestamp'].dt.floor('H').isin(hourly_anomalies)
    
    # Also look for weird transaction amounts
    if method == 'zscore':
        # Flag transactions with unusually high/low amounts
        amount_mean = completed_tx['amount'].mean()
        amount_std = completed_tx['amount'].std()
        df['amount_anomaly'] = (abs(df['amount'] - amount_mean) > threshold * amount_std) & (df['funnel_stage'] == 'completed')
    else:
        # IQR version for amounts
        Q1 = completed_tx['amount'].quantile(0.25)
        Q3 = completed_tx['amount'].quantile(0.75)
        IQR = Q3 - Q1
        df['amount_anomaly'] = ((df['amount'] < (Q1 - threshold * IQR)) | 
                                (df['amount'] > (Q3 + threshold * IQR))) & (df['funnel_stage'] == 'completed')
    
    # Create a single flag for "this is weird" combining all methods
    df['is_anomaly_detected'] = df['hourly_anomaly'] | df['amount_anomaly'] | (df['is_anomaly'] == 1)
    
    return df

def analyze_by_segment(df):
    """Analyze payment performance by different segments"""
    
    # How well each payment method performs
    payment_method_success = df[df['funnel_stage'] == 'completed'].groupby('payment_method')['status'].apply(
        lambda x: (x == 'Success').mean() * 100).reset_index()
    payment_method_success.columns = ['payment_method', 'success_rate']
    
    # Success rates by device
    device_success = df[df['funnel_stage'] == 'completed'].groupby('device_type')['status'].apply(
        lambda x: (x == 'Success').mean() * 100).reset_index()
    device_success.columns = ['device_type', 'success_rate']
    
    # Success rates by country
    country_success = df[df['funnel_stage'] == 'completed'].groupby('country')['status'].apply(
        lambda x: (x == 'Success').mean() * 100).reset_index()
    country_success.columns = ['country', 'success_rate']
    
    return {
        'payment_method_success': payment_method_success,
        'device_success': device_success,
        'country_success': country_success
    }

def create_time_series_data(df):
    """Create time series data for dashboard visualizations"""
    
    # Group by hour to see volume trends
    df['hour'] = df['timestamp'].dt.floor('H')
    hourly_volume = df[df['funnel_stage'] == 'completed'].groupby('hour')['transaction_id'].count().reset_index()
    hourly_volume.columns = ['timestamp', 'transaction_count']
    
    # Success rate by hour
    hourly_success = df[df['funnel_stage'] == 'completed'].groupby('hour')['status'].apply(
        lambda x: (x == 'Success').mean() * 100).reset_index()
    hourly_success.columns = ['timestamp', 'success_rate']
    
    # Daily progression through the funnel
    df['date'] = df['timestamp'].dt.date
    daily_funnel = df.groupby(['date', 'funnel_stage'])['transaction_id'].nunique().reset_index()
    
    return {
        'hourly_volume': hourly_volume,
        'hourly_success': hourly_success,
        'daily_funnel': daily_funnel
    }

def save_analysis_results(funnel_data, anomaly_df, segment_analysis, time_series_data, output_db='analysis_results.db'):
    """Save analysis results to SQLite for dashboard use"""
    conn = sqlite3.connect(output_db)
    
    # Package up the funnel data
    funnel_df = pd.DataFrame({
        'stage': funnel_data['stages'],
        'count': funnel_data['counts']
    })
    
    # Add drop-off rates to funnel data
    drop_off_rates = funnel_data['drop_off_rates'] + [0]  # Last stage can't have drop-off
    funnel_df['drop_off_rate'] = drop_off_rates
    
    funnel_df.to_sql('funnel_analysis', conn, if_exists='replace', index=False)
    
    # Save just the anomalies, not the whole dataset
    anomalies = anomaly_df[anomaly_df['is_anomaly_detected']].copy()
    anomalies.to_sql('anomalies', conn, if_exists='replace', index=False)
    
    # Save all segment analysis
    segment_analysis['payment_method_success'].to_sql('payment_method_performance', conn, if_exists='replace', index=False)
    segment_analysis['device_success'].to_sql('device_performance', conn, if_exists='replace', index=False)
    segment_analysis['country_success'].to_sql('country_performance', conn, if_exists='replace', index=False)
    
    # Save time series data for charts
    time_series_data['hourly_volume'].to_sql('hourly_volume', conn, if_exists='replace', index=False)
    time_series_data['hourly_success'].to_sql('hourly_success', conn, if_exists='replace', index=False)
    time_series_data['daily_funnel'].to_sql('daily_funnel', conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"Analysis results saved to {output_db}")

def main():
    # Load all transaction data
    df = load_data()
    
    # Check the payment funnel health
    funnel_data = analyze_conversion_funnel(df)
    print("\nConversion Funnel Analysis:")
    for i, stage in enumerate(funnel_data['stages']):
        count = funnel_data['counts'][i]
        drop_rate = funnel_data['drop_off_rates'][i] if i < len(funnel_data['drop_off_rates']) else 0
        print(f"{stage}: {count} transactions, {drop_rate:.2f}% drop-off")
    
    # Look for weird transactions
    anomaly_df = detect_anomalies(df, method='zscore', threshold=2.5)
    anomaly_count = anomaly_df['is_anomaly_detected'].sum()
    print(f"\nDetected {anomaly_count} anomalous transactions")
    
    # See how different segments are performing
    segment_analysis = analyze_by_segment(df)
    print("\nPayment Method Success Rates:")
    print(segment_analysis['payment_method_success'])
    
    # Create time series data for charts
    time_series_data = create_time_series_data(df)
    
    # Save everything for the dashboard to use
    save_analysis_results(funnel_data, anomaly_df, segment_analysis, time_series_data)

if __name__ == "__main__":
    main()
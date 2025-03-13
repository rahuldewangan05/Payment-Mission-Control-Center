import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

# def analyze_ab_tests(df):
#     """
#     Analyze A/B test results with statistical significance testing
    
#     Parameters:
#     -----------
#     df : DataFrame
#         Transaction data with A/B test variant information
    
#     Returns:
#     --------
#     Dictionary containing A/B test results
#     """
#     import scipy.stats as stats
    
#     # Only include completed transactions for conversion analysis
#     completed_df = df[df['funnel_stage'] == 'completed'].copy()
    
#     # Get unique transaction IDs to avoid double-counting
#     unique_txs = df.drop_duplicates('transaction_id')
    
#     # Track results for each A/B test
#     ab_test_results = {}
    
#     # Define the A/B tests to analyze
#     ab_tests = {
#         'payment_form_test': ['control', 'simplified_form'],
#         'button_color_test': ['blue_button', 'green_button'],
#         'checkout_flow_test': ['standard', 'one_page']
#     }
    
#     # Analyze each A/B test
#     for test_name, variants in ab_tests.items():
#         # Initialize results dictionary for this test
#         test_results = {
#             'variants': variants,
#             'sample_sizes': [],
#             'conversion_rates': [],
#             'p_value': None,
#             'significant': False,
#             'relative_improvement': None
#         }
        
#         # Calculate conversion rates for each variant
#         conversion_data = []
        
#         for variant in variants:
#             # Get transactions for this variant
#             variant_txs = unique_txs[unique_txs[test_name] == variant]
#             total_variant_txs = len(variant_txs)
            
#             # Get completed transactions for this variant
#             completed_variant_txs = completed_df[completed_df['transaction_id'].isin(variant_txs['transaction_id'])]
#             completed_variant_txs = completed_variant_txs[completed_variant_txs['status'] == 'Success']
#             completed_count = len(completed_variant_txs)
            
#             # Calculate conversion rate
#             conversion_rate = (completed_count / total_variant_txs) * 100 if total_variant_txs > 0 else 0
            
#             # Store data for statistical testing
#             conversion_data.append({
#                 'variant': variant,
#                 'total': total_variant_txs,
#                 'converted': completed_count,
#                 'conversion_rate': conversion_rate
#             })
            
#             # Add to results
#             test_results['sample_sizes'].append(total_variant_txs)
#             test_results['conversion_rates'].append(conversion_rate)
        
#         # Calculate statistical significance (chi-square test)
#         if len(variants) == 2 and all(cd['total'] > 0 for cd in conversion_data):
#             # Create contingency table [converted, not_converted] for each variant
#             contingency_table = [
#                 [cd['converted'], cd['total'] - cd['converted']]
#                 for cd in conversion_data
#             ]
            
#             # Perform chi-square test
#             chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            
#             # Check if result is statistically significant (p < 0.05)
#             significant = p_value < 0.05
            
#             # Calculate relative improvement (variant B vs variant A)
#             if conversion_data[0]['conversion_rate'] > 0:
#                 relative_improvement = ((conversion_data[1]['conversion_rate'] - conversion_data[0]['conversion_rate']) 
#                                         / conversion_data[0]['conversion_rate']) * 100
#             else:
#                 relative_improvement = 0
            
#             # Add to results
#             test_results['p_value'] = p_value
#             test_results['significant'] = significant
#             test_results['relative_improvement'] = relative_improvement
        
#         # Add to overall results
#         ab_test_results[test_name] = test_results
    
#     # Analyze performance by funnel stage for each variant
#     funnel_stage_analysis = {}
    
#     for test_name, variants in ab_tests.items():
#         funnel_stage_data = {}
        
#         for variant in variants:
#             # Get transactions for this variant
#             variant_df = df[df[test_name] == variant]
            
#             # Calculate drop-off at each funnel stage
#             funnel_stages = ['initiated', 'details_entered', 'otp_verification', 'processing', 'completed']
#             stage_counts = []
            
#             for stage in funnel_stages:
#                 # Count unique transactions that reached this stage
#                 stage_count = variant_df[variant_df['funnel_stage'] == stage]['transaction_id'].nunique()
#                 stage_counts.append(stage_count)
            
#             # Calculate drop-off rates between stages
#             drop_off_rates = []
#             for i in range(len(stage_counts) - 1):
#                 if stage_counts[i] > 0:
#                     drop_rate = (stage_counts[i] - stage_counts[i+1]) / stage_counts[i] * 100
#                 else:
#                     drop_rate = 0
#                 drop_off_rates.append(drop_rate)
            
#             # Store in results
#             funnel_stage_data[variant] = {
#                 'counts': stage_counts,
#                 'drop_off_rates': drop_off_rates
#             }
        
#         funnel_stage_analysis[test_name] = funnel_stage_data
    
#     # Add funnel stage analysis to results
#     ab_test_results['funnel_stage_analysis'] = funnel_stage_analysis
    
#     return ab_test_results


def analyze_ab_tests(df):
    """
    Analyze A/B test results with statistical significance testing
    
    Parameters:
    -----------
    df : DataFrame
        Transaction data with A/B test variant information
    
    Returns:
    --------
    Dictionary containing A/B test results
    """
    import scipy.stats as stats
    
    # Map expected test names to actual column names in the dataset
    test_column_mapping = {
        'payment_form_test': 'payment_form_variant',
        'button_color_test': 'button_color_variant',
        'checkout_flow_test': 'checkout_flow_variant'
    }
    
    # Only include completed transactions for conversion analysis
    completed_df = df[df['funnel_stage'] == 'completed'].copy()
    
    # Get unique transaction IDs to avoid double-counting
    unique_txs = df.drop_duplicates('transaction_id')
    
    # Track results for each A/B test
    ab_test_results = {}
    
    # Define the A/B tests to analyze with expected variant values
    ab_tests = {
        'payment_form_test': ['control', 'simplified_form'],
        'button_color_test': ['blue_button', 'green_button'],
        'checkout_flow_test': ['standard', 'one_page']
    }
    
    # Filter to only include tests that exist in the dataframe
    existing_tests = {}
    for test_name, variants in ab_tests.items():
        column_name = test_column_mapping.get(test_name)
        if column_name in df.columns:
            # Find the actual variant values in the data
            actual_variants = df[column_name].unique().tolist()
            if len(actual_variants) > 0:
                existing_tests[test_name] = {
                    'column_name': column_name,
                    'variants': actual_variants
                }
    
    if not existing_tests:
        print("Warning: No A/B test columns found in the dataset")
        return {"no_tests_found": True}
    
    # Analyze each A/B test
    for test_name, test_info in existing_tests.items():
        column_name = test_info['column_name']
        variants = test_info['variants']
        
        # Skip if we don't have at least 2 variants
        if len(variants) < 2:
            continue
        
        # Initialize results dictionary for this test
        test_results = {
            'variants': variants,
            'sample_sizes': [],
            'conversion_rates': [],
            'p_value': None,
            'significant': False,
            'relative_improvement': None
        }
        
        # Calculate conversion rates for each variant
        conversion_data = []
        
        for variant in variants:
            # Get transactions for this variant
            variant_txs = unique_txs[unique_txs[column_name] == variant]
            total_variant_txs = len(variant_txs)
            
            # Get completed transactions for this variant
            completed_variant_txs = completed_df[completed_df['transaction_id'].isin(variant_txs['transaction_id'])]
            completed_variant_txs = completed_variant_txs[completed_variant_txs['status'] == 'Success']
            completed_count = len(completed_variant_txs)
            
            # Calculate conversion rate
            conversion_rate = (completed_count / total_variant_txs) * 100 if total_variant_txs > 0 else 0
            
            # Store data for statistical testing
            conversion_data.append({
                'variant': variant,
                'total': total_variant_txs,
                'converted': completed_count,
                'conversion_rate': conversion_rate
            })
            
            # Add to results
            test_results['sample_sizes'].append(total_variant_txs)
            test_results['conversion_rates'].append(conversion_rate)
        
        # Calculate statistical significance (chi-square test)
        if len(variants) == 2 and all(cd['total'] > 0 for cd in conversion_data):
            # Create contingency table [converted, not_converted] for each variant
            contingency_table = [
                [cd['converted'], cd['total'] - cd['converted']]
                for cd in conversion_data
            ]
            
            # Perform chi-square test
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            
            # Check if result is statistically significant (p < 0.05)
            significant = p_value < 0.05
            
            # Calculate relative improvement (variant B vs variant A)
            if conversion_data[0]['conversion_rate'] > 0:
                relative_improvement = ((conversion_data[1]['conversion_rate'] - conversion_data[0]['conversion_rate']) 
                                        / conversion_data[0]['conversion_rate']) * 100
            else:
                relative_improvement = 0
            
            # Add to results
            test_results['p_value'] = p_value
            test_results['significant'] = significant
            test_results['relative_improvement'] = relative_improvement
        
        # Add to overall results
        ab_test_results[test_name] = test_results
    
    # Analyze performance by funnel stage for each variant
    funnel_stage_analysis = {}
    
    for test_name, test_info in existing_tests.items():
        column_name = test_info['column_name']
        variants = test_info['variants']
        funnel_stage_data = {}
        
        for variant in variants:
            # Get transactions for this variant
            variant_df = df[df[column_name] == variant]
            
            # Calculate drop-off at each funnel stage
            funnel_stages = ['initiated', 'details_entered', 'otp_verification', 'processing', 'completed']
            stage_counts = []
            
            for stage in funnel_stages:
                # Count unique transactions that reached this stage
                stage_count = variant_df[variant_df['funnel_stage'] == stage]['transaction_id'].nunique()
                stage_counts.append(stage_count)
            
            # Calculate drop-off rates between stages
            drop_off_rates = []
            for i in range(len(stage_counts) - 1):
                if stage_counts[i] > 0:
                    drop_rate = (stage_counts[i] - stage_counts[i+1]) / stage_counts[i] * 100
                else:
                    drop_rate = 0
                drop_off_rates.append(drop_rate)
            
            # Store in results
            funnel_stage_data[variant] = {
                'counts': stage_counts,
                'drop_off_rates': drop_off_rates
            }
        
        funnel_stage_analysis[test_name] = funnel_stage_data
    
    # Add funnel stage analysis to results
    ab_test_results['funnel_stage_analysis'] = funnel_stage_analysis
    
    return ab_test_results


# def analyze_ab_tests(df):
#     """
#     Analyze A/B test results with statistical significance testing
    
#     Parameters:
#     -----------
#     df : DataFrame
#         Transaction data with A/B test variant information
    
#     Returns:
#     --------
#     Dictionary containing A/B test results
#     """
#     import scipy.stats as stats
    
#     # Print all column names to help diagnose missing columns
#     print("Available columns in dataset:", df.columns.tolist())
    
#     # Only include completed transactions for conversion analysis
#     completed_df = df[df['funnel_stage'] == 'completed'].copy()
    
#     # Get unique transaction IDs to avoid double-counting
#     unique_txs = df.drop_duplicates('transaction_id')
    
#     # Track results for each A/B test
#     ab_test_results = {}
    
#     # Define the A/B tests to analyze
#     ab_tests = {
#         'payment_form_test': ['control', 'simplified_form'],
#         'button_color_test': ['blue_button', 'green_button'],
#         'checkout_flow_test': ['standard', 'one_page']
#     }
    
#     # Filter to only include tests that exist in the dataframe
#     existing_tests = {test: variants for test, variants in ab_tests.items() if test in df.columns}
    
#     if not existing_tests:
#         print("Warning: No A/B test columns found in the dataset")
#         return {"no_tests_found": True}
    
#     # Analyze each A/B test
#     for test_name, variants in existing_tests.items():
#         # Initialize results dictionary for this test
#         test_results = {
#             'variants': variants,
#             'sample_sizes': [],
#             'conversion_rates': [],
#             'p_value': None,
#             'significant': False,
#             'relative_improvement': None
#         }
        
#         # Calculate conversion rates for each variant
#         conversion_data = []
        
#         for variant in variants:
#             # Get transactions for this variant
#             variant_txs = unique_txs[unique_txs[test_name] == variant]
#             total_variant_txs = len(variant_txs)
            
#             # Get completed transactions for this variant
#             completed_variant_txs = completed_df[completed_df['transaction_id'].isin(variant_txs['transaction_id'])]
#             completed_variant_txs = completed_variant_txs[completed_variant_txs['status'] == 'Success']
#             completed_count = len(completed_variant_txs)
            
#             # Calculate conversion rate
#             conversion_rate = (completed_count / total_variant_txs) * 100 if total_variant_txs > 0 else 0
            
#             # Store data for statistical testing
#             conversion_data.append({
#                 'variant': variant,
#                 'total': total_variant_txs,
#                 'converted': completed_count,
#                 'conversion_rate': conversion_rate
#             })
            
#             # Add to results
#             test_results['sample_sizes'].append(total_variant_txs)
#             test_results['conversion_rates'].append(conversion_rate)
        
#         # Calculate statistical significance (chi-square test)
#         if len(variants) == 2 and all(cd['total'] > 0 for cd in conversion_data):
#             # Create contingency table [converted, not_converted] for each variant
#             contingency_table = [
#                 [cd['converted'], cd['total'] - cd['converted']]
#                 for cd in conversion_data
#             ]
            
#             # Perform chi-square test
#             chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            
#             # Check if result is statistically significant (p < 0.05)
#             significant = p_value < 0.05
            
#             # Calculate relative improvement (variant B vs variant A)
#             if conversion_data[0]['conversion_rate'] > 0:
#                 relative_improvement = ((conversion_data[1]['conversion_rate'] - conversion_data[0]['conversion_rate']) 
#                                         / conversion_data[0]['conversion_rate']) * 100
#             else:
#                 relative_improvement = 0
            
#             # Add to results
#             test_results['p_value'] = p_value
#             test_results['significant'] = significant
#             test_results['relative_improvement'] = relative_improvement
        
#         # Add to overall results
#         ab_test_results[test_name] = test_results
    
#     # Analyze performance by funnel stage for each variant
#     funnel_stage_analysis = {}
    
#     for test_name, variants in existing_tests.items():
#         funnel_stage_data = {}
        
#         for variant in variants:
#             # Get transactions for this variant
#             variant_df = df[df[test_name] == variant]
            
#             # Calculate drop-off at each funnel stage
#             funnel_stages = ['initiated', 'details_entered', 'otp_verification', 'processing', 'completed']
#             stage_counts = []
            
#             for stage in funnel_stages:
#                 # Count unique transactions that reached this stage
#                 stage_count = variant_df[variant_df['funnel_stage'] == stage]['transaction_id'].nunique()
#                 stage_counts.append(stage_count)
            
#             # Calculate drop-off rates between stages
#             drop_off_rates = []
#             for i in range(len(stage_counts) - 1):
#                 if stage_counts[i] > 0:
#                     drop_rate = (stage_counts[i] - stage_counts[i+1]) / stage_counts[i] * 100
#                 else:
#                     drop_rate = 0
#                 drop_off_rates.append(drop_rate)
            
#             # Store in results
#             funnel_stage_data[variant] = {
#                 'counts': stage_counts,
#                 'drop_off_rates': drop_off_rates
#             }
        
#         funnel_stage_analysis[test_name] = funnel_stage_data
    
#     # Add funnel stage analysis to results
#     ab_test_results['funnel_stage_analysis'] = funnel_stage_analysis
    
#     return ab_test_results

def load_data(db_path='payment_data.db'):
    """Load transaction data from SQLite database"""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM transactions"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def analyze_conversion_funnel(df):
    """Analyze the conversion funnel to identify drop-off points"""
    
    # Count unique transactions at each funnel stage
    funnel_data = df.groupby('funnel_stage')['transaction_id'].nunique().reset_index()
    
    # Calculate drop-off rates
    funnel_stages = ['initiated', 'details_entered', 'otp_verification', 'processing', 'completed']
    funnel_counts = []
    for stage in funnel_stages:
        count = funnel_data[funnel_data['funnel_stage'] == stage]['transaction_id'].values
        funnel_counts.append(count[0] if len(count) > 0 else 0)
    
    # Calculate drop-off rates
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
    # Get only completed transactions for anomaly detection
    completed_tx = df[df['funnel_stage'] == 'completed'].copy()
    
    # Group by hour to check for unusual patterns
    completed_tx['hour'] = completed_tx['timestamp'].dt.floor('H')
    hourly_counts = completed_tx.groupby('hour')['transaction_id'].count()
    
    # Apply anomaly detection
    if method == 'zscore':
        # Z-score method
        z_scores = np.abs(stats.zscore(hourly_counts))
        hourly_anomalies = hourly_counts[z_scores > threshold].index.tolist()
    else:
        # IQR method
        Q1 = hourly_counts.quantile(0.25)
        Q3 = hourly_counts.quantile(0.75)
        IQR = Q3 - Q1
        hourly_anomalies = hourly_counts[(hourly_counts < (Q1 - threshold * IQR)) | 
                                        (hourly_counts > (Q3 + threshold * IQR))].index.tolist()
    
    # Mark transactions in anomalous hours
    df['hourly_anomaly'] = df['timestamp'].dt.floor('H').isin(hourly_anomalies)
    
    # Additionally, detect anomalies in transaction amounts for completed transactions
    if method == 'zscore':
        # Z-score on transaction amounts
        amount_mean = completed_tx['amount'].mean()
        amount_std = completed_tx['amount'].std()
        df['amount_anomaly'] = (abs(df['amount'] - amount_mean) > threshold * amount_std) & (df['funnel_stage'] == 'completed')
    else:
        # IQR on transaction amounts
        Q1 = completed_tx['amount'].quantile(0.25)
        Q3 = completed_tx['amount'].quantile(0.75)
        IQR = Q3 - Q1
        df['amount_anomaly'] = ((df['amount'] < (Q1 - threshold * IQR)) | 
                                (df['amount'] > (Q3 + threshold * IQR))) & (df['funnel_stage'] == 'completed')
    
    # Combined anomaly flag
    df['is_anomaly_detected'] = df['hourly_anomaly'] | df['amount_anomaly'] | (df['is_anomaly'] == 1)
    
    return df

def analyze_by_segment(df):
    """Analyze payment performance by different segments"""
    
    # Success rate by payment method
    payment_method_success = df[df['funnel_stage'] == 'completed'].groupby('payment_method')['status'].apply(
        lambda x: (x == 'Success').mean() * 100).reset_index()
    payment_method_success.columns = ['payment_method', 'success_rate']
    
    # Success rate by device type
    device_success = df[df['funnel_stage'] == 'completed'].groupby('device_type')['status'].apply(
        lambda x: (x == 'Success').mean() * 100).reset_index()
    device_success.columns = ['device_type', 'success_rate']
    
    # Success rate by country
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
    
    # Hourly transaction volume
    df['hour'] = df['timestamp'].dt.floor('h')
    hourly_volume = df[df['funnel_stage'] == 'completed'].groupby('hour')['transaction_id'].count().reset_index()
    hourly_volume.columns = ['timestamp', 'transaction_count']
    
    # Hourly success rate
    hourly_success = df[df['funnel_stage'] == 'completed'].groupby('hour')['status'].apply(
        lambda x: (x == 'Success').mean() * 100).reset_index()
    hourly_success.columns = ['timestamp', 'success_rate']
    
    # Daily funnel progression
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
    
    # Save funnel data
    funnel_df = pd.DataFrame({
        'stage': funnel_data['stages'],
        'count': funnel_data['counts']
    })
    
    # Add drop-off rates (one less than stages)
    drop_off_rates = funnel_data['drop_off_rates'] + [0]  # Add 0 for the last stage
    funnel_df['drop_off_rate'] = drop_off_rates
    
    funnel_df.to_sql('funnel_analysis', conn, if_exists='replace', index=False)
    
    # Save anomaly data (only the anomalies)
    anomalies = anomaly_df[anomaly_df['is_anomaly_detected']].copy()
    anomalies.to_sql('anomalies', conn, if_exists='replace', index=False)
    
    # Save segment analysis
    segment_analysis['payment_method_success'].to_sql('payment_method_performance', conn, if_exists='replace', index=False)
    segment_analysis['device_success'].to_sql('device_performance', conn, if_exists='replace', index=False)
    segment_analysis['country_success'].to_sql('country_performance', conn, if_exists='replace', index=False)
    
    # Save time series data
    time_series_data['hourly_volume'].to_sql('hourly_volume', conn, if_exists='replace', index=False)
    time_series_data['hourly_success'].to_sql('hourly_success', conn, if_exists='replace', index=False)
    time_series_data['daily_funnel'].to_sql('daily_funnel', conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"Analysis results saved to {output_db}")


def save_ab_test_results(ab_test_results, output_db='analysis_results.db'):
    """Save A/B test results to SQLite database"""
    conn = sqlite3.connect(output_db)
    
    # Create a DataFrame for the main A/B test results
    test_data = []
    
    for test_name, results in ab_test_results.items():
        if test_name != 'funnel_stage_analysis':
            for i, variant in enumerate(results['variants']):
                test_data.append({
                    'test_name': test_name,
                    'variant': variant,
                    'sample_size': results['sample_sizes'][i],
                    'conversion_rate': results['conversion_rates'][i],
                    'p_value': results['p_value'] if i == 1 else None,  # Only show p-value for the second variant
                    'significant': results['significant'] if i == 1 else None,
                    'relative_improvement': results['relative_improvement'] if i == 1 else None
                })
    
    # Save to database
    if test_data:
        ab_test_df = pd.DataFrame(test_data)
        ab_test_df.to_sql('ab_test_results', conn, if_exists='replace', index=False)
    
    # Save funnel stage analysis for each A/B test
    funnel_stage_data = []
    
    funnel_stages = ['initiated', 'details_entered', 'otp_verification', 'processing', 'completed']
    
    if 'funnel_stage_analysis' in ab_test_results:
        for test_name, variants_data in ab_test_results['funnel_stage_analysis'].items():
            for variant, data in variants_data.items():
                for i, stage in enumerate(funnel_stages):
                    # Add stage counts
                    funnel_stage_data.append({
                        'test_name': test_name,
                        'variant': variant,
                        'funnel_stage': stage,
                        'count': data['counts'][i],
                        'drop_off_rate': data['drop_off_rates'][i] if i < len(data['drop_off_rates']) else None
                    })
    
    # Save to database
    if funnel_stage_data:
        funnel_stage_df = pd.DataFrame(funnel_stage_data)
        funnel_stage_df.to_sql('ab_test_funnel_analysis', conn, if_exists='replace', index=False)
    
    conn.close()


def main():
    # Load data
    df = load_data()
    
    # Analyze conversion funnel
    funnel_data = analyze_conversion_funnel(df)
    print("\nConversion Funnel Analysis:")
    for i, stage in enumerate(funnel_data['stages']):
        count = funnel_data['counts'][i]
        drop_rate = funnel_data['drop_off_rates'][i] if i < len(funnel_data['drop_off_rates']) else 0
        print(f"{stage}: {count} transactions, {drop_rate:.2f}% drop-off")
    
    # Detect anomalies
    anomaly_df = detect_anomalies(df, method='zscore', threshold=2.5)
    anomaly_count = anomaly_df['is_anomaly_detected'].sum()
    print(f"\nDetected {anomaly_count} anomalous transactions")
    
    # Analyze by segment
    segment_analysis = analyze_by_segment(df)
    print("\nPayment Method Success Rates:")
    print(segment_analysis['payment_method_success'])
    
    # Create time series data
    time_series_data = create_time_series_data(df)
    
    # Save results for dashboard
    save_analysis_results(funnel_data, anomaly_df, segment_analysis, time_series_data)
    
    # # Analyze A/B tests
    # ab_test_results = analyze_ab_tests(df)
    
    # # Print A/B test results
    # print("\nA/B Test Results:")
    # for test_name, results in ab_test_results.items():
    #     if test_name != 'funnel_stage_analysis':
    #         print(f"\n{test_name}:")
    #         for i, variant in enumerate(results['variants']):
    #             print(f"  {variant}: {results['conversion_rates'][i]:.2f}% conversion rate (n={results['sample_sizes'][i]})")
            
    #         if results['p_value'] is not None:
    #             print(f"  p-value: {results['p_value']:.4f} {'(Significant)' if results['significant'] else '(Not significant)'}")
    #             print(f"  Relative improvement: {results['relative_improvement']:.2f}%")
    
    # # Save A/B test results to the analysis database
    # save_ab_test_results(ab_test_results, 'analysis_results.db')
    
    # Analyze A/B tests
    ab_test_results = analyze_ab_tests(df)
    
    # Print A/B test results
    if "no_tests_found" in ab_test_results:
        print("\nNo A/B test columns found in the dataset")
    else:
        print("\nA/B Test Results:")
        for test_name, results in ab_test_results.items():
            if test_name != 'funnel_stage_analysis':
                print(f"\n{test_name}:")
                for i, variant in enumerate(results['variants']):
                    print(f"  {variant}: {results['conversion_rates'][i]:.2f}% conversion rate (n={results['sample_sizes'][i]})")
                
                if results['p_value'] is not None:
                    print(f"  p-value: {results['p_value']:.4f} {'(Significant)' if results['significant'] else '(Not significant)'}")
                    print(f"  Relative improvement: {results['relative_improvement']:.2f}%")
    
    # Save A/B test results to the analysis database only if we have results
    if "no_tests_found" not in ab_test_results:
        save_ab_test_results(ab_test_results, 'analysis_results.db')

if __name__ == "__main__":
    main()
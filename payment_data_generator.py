import pandas as pd
import numpy as np
import sqlite3
import datetime
import random
from faker import Faker

# Initialize Faker for generating realistic user IDs
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_payment_data(num_transactions=5000):
    """Generate synthetic payment transaction data with A/B test variants"""
    
    # Create unique user IDs
    user_ids = [fake.uuid4() for _ in range(1000)]
    
    # Create transaction IDs
    transaction_ids = [fake.uuid4() for _ in range(num_transactions)]
    
    # Generate timestamps over the last 30 days
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
    timestamps = [start_date + (end_date - start_date) * random.random() for _ in range(num_transactions)]
    timestamps.sort()  # Sort timestamps in chronological order
    
    # Generate transaction amounts (most between $10-$200 with some outliers)
    amounts = np.random.lognormal(mean=4.0, sigma=0.7, size=num_transactions)
    
    # Define the funnel stages
    funnel_stages = ['initiated', 'details_entered', 'otp_verification', 'processing', 'completed']
    
    # A/B Test Definitions
    ab_test_variants = {
        'payment_form_test': ['control', 'simplified_form'],
        'button_color_test': ['blue_button', 'green_button'],
        'checkout_flow_test': ['standard', 'one_page']
    }
    
    # Assign users to test variants (fixed assignment per user)
    user_test_assignments = {}
    for user_id in user_ids:
        user_test_assignments[user_id] = {
            test_name: random.choice(variants)
            for test_name, variants in ab_test_variants.items()
        }
    
    # Initialize data structures
    data = {
        'transaction_id': [],
        'user_id': [],
        'amount': [],
        'timestamp': [],
        'funnel_stage': [],
        'status': [],
        'payment_method': [],
        'device_type': [],
        'country': [],
        'is_anomaly': [],
        # A/B test fields
        'payment_form_variant': [],
        'button_color_variant': [],
        'checkout_flow_variant': []
    }
    
    # Generate payment methods
    payment_methods = ['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet', 'crypto']
    payment_method_weights = [0.4, 0.3, 0.1, 0.15, 0.05]
    
    # Generate device types
    device_types = ['mobile', 'desktop', 'tablet']
    device_type_weights = [0.6, 0.3, 0.1]
    
    # Generate countries (mostly US with some others)
    countries = ['US', 'CA', 'UK', 'DE', 'FR', 'IN', 'AU', 'JP']
    country_weights = [0.7, 0.05, 0.05, 0.04, 0.04, 0.05, 0.04, 0.03]
    
    # Create some user segments with different success rates
    # Regular users
    regular_users = user_ids[:700]
    # New users (higher failure rate)
    new_users = user_ids[700:850]
    # Premium users (lower failure rate)
    premium_users = user_ids[850:]
    
    # Funnel drop-off rates (percentage of users who drop at each stage)
    # Format: {stage: drop_off_percentage}
    normal_drop_rates = {
        'initiated': 0.05,
        'details_entered': 0.15,
        'otp_verification': 0.10,
        'processing': 0.05
    }
    
    new_user_drop_rates = {
        'initiated': 0.15,
        'details_entered': 0.25,
        'otp_verification': 0.20,
        'processing': 0.10
    }
    
    premium_user_drop_rates = {
        'initiated': 0.02,
        'details_entered': 0.05,
        'otp_verification': 0.03,
        'processing': 0.01
    }
    
    # A/B test effect on drop-off rates
    variant_effects = {
        'payment_form_test': {
            'control': {'details_entered': 0},  # No change for control
            'simplified_form': {'details_entered': -0.05}  # 5% lower drop-off
        },
        'button_color_test': {
            'blue_button': {'initiated': 0},  # No change for control
            'green_button': {'initiated': -0.02}  # 2% lower drop-off
        },
        'checkout_flow_test': {
            'standard': {'otp_verification': 0},  # No change for control
            'one_page': {'otp_verification': -0.03}  # 3% lower drop-off
        }
    }
    
    # Generate transaction data
    for i in range(num_transactions):
        transaction_id = transaction_ids[i]
        user_id = random.choice(user_ids)
        amount = round(amounts[i], 2)
        timestamp = timestamps[i]
        payment_method = random.choices(payment_methods, weights=payment_method_weights)[0]
        device_type = random.choices(device_types, weights=device_type_weights)[0]
        country = random.choices(countries, weights=country_weights)[0]
        
        # Get A/B test variants for this user
        payment_form_variant = user_test_assignments[user_id]['payment_form_test']
        button_color_variant = user_test_assignments[user_id]['button_color_test']
        checkout_flow_variant = user_test_assignments[user_id]['checkout_flow_test']
        
        # Determine drop-off rates based on user segment
        if user_id in new_users:
            drop_rates = new_user_drop_rates.copy()
        elif user_id in premium_users:
            drop_rates = premium_user_drop_rates.copy()
        else:
            drop_rates = normal_drop_rates.copy()
        
        # Apply variant effects to drop-off rates
        for test_name, variants in variant_effects.items():
            user_variant = user_test_assignments[user_id][test_name]
            effects = variants[user_variant]
            for stage, effect in effects.items():
                drop_rates[stage] = max(0, drop_rates[stage] + effect)  # Ensure no negative drop rate
        
        # Determine how far this transaction progresses in the funnel
        final_stage_reached = 'completed'
        for stage in funnel_stages[:-1]:  # Exclude 'completed'
            if random.random() < drop_rates.get(stage, 0):
                final_stage_reached = stage
                break
        
        # Add records for each stage up to the final stage
        stage_index = funnel_stages.index(final_stage_reached)
        for j in range(stage_index + 1):
            stage = funnel_stages[j]
            
            # Status is success unless it's the final stage and not completed
            status = 'Success'
            if stage == final_stage_reached and final_stage_reached != 'completed':
                status = 'Failure'
            
            # Create deliberate anomalies (about 2% of transactions)
            is_anomaly = False
            if stage == 'completed' and random.random() < 0.02:
                # Anomalies in completed transactions
                amount = round(random.uniform(1000, 5000), 2)  # Unusually high amount
                is_anomaly = True
            
            # Add a time delay for each stage (0-2 minutes between stages)
            stage_timestamp = timestamp + datetime.timedelta(minutes=j * random.uniform(0, 2))
            
            data['transaction_id'].append(transaction_id)
            data['user_id'].append(user_id)
            data['amount'].append(amount)
            data['timestamp'].append(stage_timestamp)
            data['funnel_stage'].append(stage)
            data['status'].append(status)
            data['payment_method'].append(payment_method)
            data['device_type'].append(device_type)
            data['country'].append(country)
            data['is_anomaly'].append(int(is_anomaly))
            data['payment_form_variant'].append(payment_form_variant)
            data['button_color_variant'].append(button_color_variant)
            data['checkout_flow_variant'].append(checkout_flow_variant)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert timestamp to string for SQLite compatibility
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df

def save_to_sqlite(df, db_name='payment_data.db'):
    """Save DataFrame to SQLite database"""
    conn = sqlite3.connect(db_name)
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Data saved to {db_name}")

if __name__ == "__main__":
    # Generate 5000 transactions
    payment_data = generate_payment_data(5000)
    
    # Display sample data
    print(payment_data.head())
    
    # Get statistics
    print("\nData Statistics:")
    print(f"Total number of transactions: {payment_data['transaction_id'].nunique()}")
    print(f"Total number of users: {payment_data['user_id'].nunique()}")
    print(f"Funnel stages: {payment_data['funnel_stage'].unique()}")
    print(f"Status distribution: \n{payment_data['status'].value_counts()}")
    
    # A/B test distribution stats
    print("\nA/B Test Distribution:")
    for test_col in ['payment_form_variant', 'button_color_variant', 'checkout_flow_variant']:
        print(f"{test_col}: \n{payment_data[test_col].value_counts(normalize=True)}")
    
    # Save to SQLite database
    save_to_sqlite(payment_data)
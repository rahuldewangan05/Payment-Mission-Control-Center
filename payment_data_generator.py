import pandas as pd
import numpy as np
import sqlite3
import datetime
import random
from faker import Faker

# Setting up Faker - this helps us make those random user IDs look legit
fake = Faker()

# Got tired of random results changing every time I run this - fixed seed
np.random.seed(42)
random.seed(42)

def generate_payment_data(num_transactions=5000):
    """Make a bunch of fake payment data that looks real enough"""
    
    # Need about 1000 different users for this to look realistic
    user_ids = [fake.uuid4() for _ in range(1000)]
    
    # Each transaction needs its own ID - duh
    transaction_ids = [fake.uuid4() for _ in range(num_transactions)]
    
    # Let's pretend these happened over the last month
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
    timestamps = [start_date + (end_date - start_date) * random.random() for _ in range(num_transactions)]
    timestamps.sort()  # Chronological order makes more sense
    
    # People usually spend between $10-$200, but some weirdos go higher
    amounts = np.random.lognormal(mean=4.0, sigma=0.7, size=num_transactions)
    
    # Typical journey for a payment - from start to finish
    funnel_stages = ['initiated', 'details_entered', 'otp_verification', 'processing', 'completed']
    
    # Columns we need for our dataset
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
        'is_anomaly': []
    }
    
    # Ways people can pay - credit cards still king here
    payment_methods = ['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet', 'crypto']
    payment_method_weights = [0.4, 0.3, 0.1, 0.15, 0.05]
    
    # Most folks on their phones these days
    device_types = ['mobile', 'desktop', 'tablet']
    device_type_weights = [0.6, 0.3, 0.1]
    
    # Mostly US traffic, sprinkling in some international
    countries = ['US', 'CA', 'UK', 'DE', 'FR', 'IN', 'AU', 'JP']
    country_weights = [0.7, 0.05, 0.05, 0.04, 0.04, 0.05, 0.04, 0.03]
    
    # Different kinds of users behave differently
    # The regulars - most of our userbase
    regular_users = user_ids[:700]
    # Newbies - still figuring things out, more likely to bail
    new_users = user_ids[700:850]
    # The VIPs - smoother sailing for these folks
    premium_users = user_ids[850:]
    
    # How many people give up at each step - percentages
    # Regular users drop-off rates
    normal_drop_rates = {
        'initiated': 0.05,
        'details_entered': 0.15,
        'otp_verification': 0.10,
        'processing': 0.05
    }
    
    # New users bail more often - this tracks with what we see in production
    new_user_drop_rates = {
        'initiated': 0.15,
        'details_entered': 0.25,
        'otp_verification': 0.20,
        'processing': 0.10
    }
    
    # Premium users stick it out more often
    premium_user_drop_rates = {
        'initiated': 0.02,
        'details_entered': 0.05,
        'otp_verification': 0.03,
        'processing': 0.01
    }
    
    # OK, let's build this monster dataset
    for i in range(num_transactions):
        transaction_id = transaction_ids[i]
        user_id = random.choice(user_ids)
        amount = round(amounts[i], 2)
        timestamp = timestamps[i]
        payment_method = random.choices(payment_methods, weights=payment_method_weights)[0]
        device_type = random.choices(device_types, weights=device_type_weights)[0]
        country = random.choices(countries, weights=country_weights)[0]
        
        # Figure out how likely this transaction will fail based on user type
        if user_id in new_users:
            drop_rates = new_user_drop_rates
        elif user_id in premium_users:
            drop_rates = premium_user_drop_rates
        else:
            drop_rates = normal_drop_rates
        
        # Roll the dice and see where this transaction stops
        final_stage_reached = 'completed'
        for stage in funnel_stages[:-1]:  # Don't check the last stage
            if random.random() < drop_rates.get(stage, 0):
                final_stage_reached = stage
                break
        
        # Add a row for each stage this transaction went through
        stage_index = funnel_stages.index(final_stage_reached)
        for j in range(stage_index + 1):
            stage = funnel_stages[j]
            
            # If it's the last stage and not completed, it failed
            status = 'Success'
            if stage == final_stage_reached and final_stage_reached != 'completed':
                status = 'Failure'
            
            # Throw in some weird transactions - fraud team needs something to do
            is_anomaly = False
            if stage == 'completed' and random.random() < 0.02:
                # Whoa, that's a big purchase! Red flag.
                amount = round(random.uniform(1000, 5000), 2)
                is_anomaly = True
            
            # People don't zip through payment forms - add some delay between steps
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
    
    # Put it all together in a nice dataframe
    df = pd.DataFrame(data)
    
    # SQLite doesn't play nice with datetime objects - convert to strings
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df

def save_to_sqlite(df, db_name='payment_data.db'):
    """Dump all this data into SQLite so we can query it later"""
    conn = sqlite3.connect(db_name)
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Data saved to {db_name}")

if __name__ == "__main__":
    # Let's make 5000 transactions - should be enough for testing
    payment_data = generate_payment_data(5000)
    
    # Take a peek at what we've created
    print(payment_data.head())
    
    # Some quick stats to sanity-check our data
    print("\nData Statistics:")
    print(f"Total number of transactions: {payment_data['transaction_id'].nunique()}")
    print(f"Total number of users: {payment_data['user_id'].nunique()}")
    print(f"Funnel stages: {payment_data['funnel_stage'].unique()}")
    print(f"Status distribution: \n{payment_data['status'].value_counts()}")
    
    # Save it for later analysis
    save_to_sqlite(payment_data)
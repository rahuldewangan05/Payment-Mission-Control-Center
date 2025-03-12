import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# Let's make this look sexy on the browser
st.set_page_config(
    page_title="Payment Mission Control Center",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Grab our data and cache it - no need to hammer the DB every 5 seconds
@st.cache_data(ttl=300)  # Refresh every 5 mins
def load_data(db_path='analysis_results.db'):
    conn = sqlite3.connect(db_path)
    
    # Pull everything we need in one go
    funnel_df = pd.read_sql_query("SELECT * FROM funnel_analysis", conn)
    anomalies_df = pd.read_sql_query("SELECT * FROM anomalies", conn)
    payment_method_df = pd.read_sql_query("SELECT * FROM payment_method_performance", conn)
    device_df = pd.read_sql_query("SELECT * FROM device_performance", conn)
    country_df = pd.read_sql_query("SELECT * FROM country_performance", conn)
    hourly_volume_df = pd.read_sql_query("SELECT * FROM hourly_volume", conn)
    hourly_success_df = pd.read_sql_query("SELECT * FROM hourly_success", conn)
    daily_funnel_df = pd.read_sql_query("SELECT * FROM daily_funnel", conn)
    
    # Fix the timestamps - SQLite doesn't handle datetimes properly
    hourly_volume_df['timestamp'] = pd.to_datetime(hourly_volume_df['timestamp'])
    hourly_success_df['timestamp'] = pd.to_datetime(hourly_success_df['timestamp'])
    daily_funnel_df['date'] = pd.to_datetime(daily_funnel_df['date'])
    
    # Do the same for anomalies if we have them
    if 'timestamp' in anomalies_df.columns:
        anomalies_df['timestamp'] = pd.to_datetime(anomalies_df['timestamp'])
    
    conn.close()
    
    # Pack everything into a nice dictionary for easy access
    return {
        'funnel': funnel_df,
        'anomalies': anomalies_df,
        'payment_method': payment_method_df,
        'device': device_df,
        'country': country_df,
        'hourly_volume': hourly_volume_df,
        'hourly_success': hourly_success_df,
        'daily_funnel': daily_funnel_df
    }

# Get raw transaction data for deep-diving
@st.cache_data(ttl=300)
def load_transaction_data(db_path='payment_data.db'):
    conn = sqlite3.connect(db_path)
    transactions_df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    
    # Fix timestamps here too
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
    
    return transactions_df

# Make our dashboard look snazzy
def create_header():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://img.icons8.com/fluency/96/control-panel.png", width=100)
    with col2:
        st.title("Payment Mission Control Center")
        st.markdown("Real-time monitoring, anomaly detection, and conversion optimization for payment systems")

# The main event - this builds our entire dashboard
def create_dashboard():
    # Try to load the data - if it fails, show a friendly error
    try:
        data = load_data()
        transactions_df = load_transaction_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please make sure to run the data generator and analysis scripts first.")
        st.stop()
    
    # Add our fancy header
    create_header()
    
    # Create tabs to organize our content
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Conversion Funnel", "Anomaly Detection", "Transaction Explorer"])
    
    with tab1:
        create_overview_tab(data, transactions_df)
    
    with tab2:
        create_funnel_tab(data)
        
    with tab3:
        create_anomaly_tab(data)
        
    with tab4:
        create_explorer_tab(transactions_df)

def create_overview_tab(data, transactions_df):
    # Show some KPIs at the top for quick insights
    st.subheader("Key Metrics")
    
    # Crunch the numbers for our KPIs
    total_transactions = transactions_df['transaction_id'].nunique()
    success_rate = transactions_df[transactions_df['funnel_stage'] == 'completed']['status'].value_counts(normalize=True).get('Success', 0) * 100
    conversion_rate = data['funnel']['count'].iloc[-1] / data['funnel']['count'].iloc[0] * 100
    avg_transaction = transactions_df[transactions_df['funnel_stage'] == 'completed']['amount'].mean()
    
    # Layout the metrics in a nice row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{total_transactions:,}")
    with col2:
        st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
    with col3:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        st.metric("Avg. Transaction", f"${avg_transaction:.2f}")
    
    # Show transaction volume over time
    st.subheader("Transaction Volume (Last 30 Days)")
    
    # Create the base transaction volume chart
    fig = px.line(
        data['hourly_volume'], 
        x='timestamp', 
        y='transaction_count',
        title="Hourly Transaction Volume",
        labels={'timestamp': 'Time', 'transaction_count': 'Transactions'}
    )
    
    # Add success rate on a second y-axis
    fig2 = px.line(
        data['hourly_success'], 
        x='timestamp', 
        y='success_rate', 
        line_shape='spline'
    )
    
    # Set the success rate to use the right y-axis and make it green
    fig2.update_traces(yaxis="y2", line=dict(color="green"))
    
    # Combine the charts
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add both sets of data to our combo chart
    fig3.add_traces(fig.data + fig2.data)
    
    # Make it pretty
    fig3.update_layout(
        yaxis_title="Transaction Count",
        yaxis2=dict(
            title=dict(
            text="Success Rate (%)",
            font=dict(color="green")
            ),
            tickfont=dict(color="green"),
            range=[50, 100]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Show payment performance by segment
    st.subheader("Payment Performance by Segment")
    
    # Side-by-side charts for payment methods and devices
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment method success rates
        fig = px.bar(
            data['payment_method'],
            x='payment_method',
            y='success_rate',
            title="Success Rate by Payment Method",
            labels={'payment_method': 'Payment Method', 'success_rate': 'Success Rate (%)'},
            color='success_rate',
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Device type success rates
        fig = px.bar(
            data['device'],
            x='device_type',
            y='success_rate',
            title="Success Rate by Device Type",
            labels={'device_type': 'Device Type', 'success_rate': 'Success Rate (%)'},
            color='success_rate',
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

def create_funnel_tab(data):
    st.subheader("Conversion Funnel Analysis")
    
    # Get our funnel data
    funnel_df = data['funnel']
    
    # Calculate percentages for the funnel chart
    funnel_df['percent'] = (funnel_df['count'] / funnel_df['count'].iloc[0] * 100).round(1)
    funnel_df['percent_label'] = funnel_df['percent'].apply(lambda x: f"{x}%")
    
    # Create the funnel visualization
    fig = go.Figure(go.Funnel(
        y=funnel_df['stage'],
        x=funnel_df['count'],
        textposition="inside",
        textinfo="value+percent initial",
        opacity=0.8,
        marker={"color": ["#0099ff", "#4db8ff", "#99d6ff", "#cceaff", "#e6f5ff"]},
        connector={"line": {"color": "royalblue", "dash": "solid", "width": 3}}
    ))
    
    fig.update_layout(
        title="Payment Conversion Funnel",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyze where people are dropping off
    st.subheader("Funnel Drop-off Analysis")
    
    # Two columns for related charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a bar chart showing drop-off rates by stage
        drop_off_df = funnel_df.copy()
        # Skip the last stage - can't drop off from completed
        drop_off_df.loc[drop_off_df.index[-1], 'drop_off_rate'] = np.nan
        
        fig = px.bar(
            drop_off_df,
            x='stage',
            y='drop_off_rate',
            title="Drop-off Rate by Stage",
            labels={'stage': 'Funnel Stage', 'drop_off_rate': 'Drop-off Rate (%)'},
            color='drop_off_rate',
            color_continuous_scale=px.colors.sequential.Reds
        )
        
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Show how completed transactions trend over time
        daily_funnel = data['daily_funnel'].copy()
        pivot_df = daily_funnel.pivot(index='date', columns='funnel_stage', values='transaction_id').reset_index()
        
        # Make sure we have data for all stages
        for stage in ['initiated', 'details_entered', 'otp_verification', 'processing', 'completed']:
            if stage not in pivot_df.columns:
                pivot_df[stage] = 0
        
        # Plot the completed stage over time
        fig = px.line(
            pivot_df, 
            x='date', 
            y='completed',
            title="Daily Completed Transactions",
            labels={'date': 'Date', 'completed': 'Completed Transactions'},
            line_shape='spline'
        )
        
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Give some recommendations based on the data
    st.subheader("Improvement Recommendations")
    
    # Find where we're bleeding the most users
    highest_dropoff_idx = funnel_df['drop_off_rate'].iloc[:-1].idxmax()
    highest_dropoff_stage = funnel_df.loc[highest_dropoff_idx, 'stage']
    highest_dropoff_rate = funnel_df.loc[highest_dropoff_idx, 'drop_off_rate']
    
    st.info(f"🔍 **Key Finding**: The highest drop-off rate ({highest_dropoff_rate:.1f}%) occurs at the **{highest_dropoff_stage}** stage.")
    
    # Tailored recommendations based on the problematic stage
    if highest_dropoff_stage == 'initiated':
        st.markdown("""
        **Recommendations to improve initial page:**
        - Simplify the payment form to reduce cognitive load
        - Add progress indicators to set expectations
        - Implement one-click payment options for returning users
        """)
    elif highest_dropoff_stage == 'details_entered':
        st.markdown("""
        **Recommendations to improve details collection:**
        - Implement auto-fill functionality
        - Add card scanning capability for mobile
        - Improve form validation with inline feedback
        - Consider implementing digital wallet integration
        """)
    elif highest_dropoff_stage == 'otp_verification':
        st.markdown("""
        **Recommendations to improve OTP verification:**
        - Streamline OTP entry with auto-focus fields
        - Add option for push notifications instead of SMS
        - Implement biometric authentication where possible
        - Optimize OTP delivery speed
        """)
    elif highest_dropoff_stage == 'processing':
        st.markdown("""
        **Recommendations to improve processing:**
        - Optimize backend processing time
        - Implement better error handling and recovery
        - Provide clearer feedback during processing
        - Consider implementing parallel processing paths
        """)

def create_anomaly_tab(data):
    st.subheader("Anomaly Detection")
    
    # Get our anomaly data
    anomalies_df = data['anomalies']
    
    # Show some key stats about anomalies
    st.subheader("Anomaly Metrics")
    
    total_anomalies = len(anomalies_df)
    
    # Display metrics in three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Anomalies Detected", f"{total_anomalies}")
    with col2:
        if 'amount' in anomalies_df.columns:
            avg_anomaly_amount = anomalies_df['amount'].mean()
            st.metric("Avg. Anomaly Amount", f"${avg_anomaly_amount:.2f}")
    with col3:
        if 'timestamp' in anomalies_df.columns:
            latest_anomaly = anomalies_df['timestamp'].max()
            st.metric("Latest Anomaly", f"{latest_anomaly.strftime('%Y-%m-%d %H:%M')}")
    
    # Show a table of recent anomalies
    st.subheader("Recent Anomalies")
    
    if len(anomalies_df) > 0:
        # Only show the important columns
        display_cols = ['transaction_id', 'amount', 'timestamp', 'funnel_stage', 'status', 'payment_method']
        display_cols = [col for col in display_cols if col in anomalies_df.columns]
        
        # Show the 10 most recent ones
        display_df = anomalies_df[display_cols].sort_values('timestamp', ascending=False).head(10)
        
        # Make timestamps pretty
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(display_df)
    else:
        st.info("No anomalies detected in the current dataset.")
    
    # Visualize where anomalies are occurring
    if 'timestamp' in anomalies_df.columns and 'amount' in anomalies_df.columns:
        st.subheader("Anomaly Visualization")
        
        # Create a chart showing normal transactions
        fig = px.scatter(
            data['hourly_volume'],
            x='timestamp',
            y='transaction_count',
            title="Transaction Volume with Anomalies",
            labels={'timestamp': 'Time', 'transaction_count': 'Transaction Count'}
        )
        
        # Overlay the anomalies as red dots
        if len(anomalies_df) > 0:
            # Group anomalies by hour to match our other data
            anomalies_df['hour'] = anomalies_df['timestamp'].dt.floor('H')
            hourly_anomalies = anomalies_df.groupby('hour').size().reset_index()
            hourly_anomalies.columns = ['timestamp', 'anomaly_count']
            
            # Add the red dots for anomalies
            fig.add_trace(
                go.Scatter(
                    x=hourly_anomalies['timestamp'],
                    y=hourly_anomalies['anomaly_count'],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Anomalies'
                )
            )
        
        st.plotly_chart(fig, use_container_width=True)

def create_explorer_tab(transactions_df):
    st.subheader("Transaction Explorer")
    
    # Let users filter transactions
    st.markdown("### Filters")
    
    # Three columns for our filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date picker for time range
        min_date = transactions_df['timestamp'].min().date()
        max_date = transactions_df['timestamp'].max().date()
        
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    with col2:
        # Transaction status filter
        status_options = ['All'] + sorted(transactions_df['status'].unique().tolist())
        selected_status = st.selectbox("Status", status_options)
    
    with col3:
        # Funnel stage filter
        stage_options = ['All'] + sorted(transactions_df['funnel_stage'].unique().tolist())
        selected_stage = st.selectbox("Funnel Stage", stage_options)
    
    # Apply all selected filters
    filtered_df = transactions_df.copy()
    
    # Date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= start_date) & 
            (filtered_df['timestamp'].dt.date <= end_date)
        ]
    
    # Status filter
    if selected_status != 'All':
        filtered_df = filtered_df[filtered_df['status'] == selected_status]
    
    # Stage filter
    if selected_stage != 'All':
        filtered_df = filtered_df[filtered_df['funnel_stage'] == selected_stage]
    
    # Show the filtered data
    st.markdown(f"### Transactions ({len(filtered_df)} records)")
    
    # Display the matching transactions
    st.dataframe(filtered_df.sort_values('timestamp', ascending=False))
    
    # Add a download button so users can analyze it offline
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="payment_transactions.csv",
        mime="text/csv"
    )

# Fire everything up when run directly
if __name__ == "__main__":
    create_dashboard()
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Payment Analytics Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 30px;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #1976D2;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .subsection-header {
        font-size: 20px;
        font-weight: bold;
        color: #1565C0;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 2px 2px 2px rgba(0,0,0,0.1);
    }
    .success-text {
        color: #388E3C;
        font-weight: bold;
    }
    .warning-text {
        color: #FFA000;
        font-weight: bold;
    }
    .danger-text {
        color: #D32F2F;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data from SQLite database
@st.cache_data(ttl=3600)
def load_data(db_path='analysis_results.db'):
    """Load analysis results from SQLite database"""
    conn = sqlite3.connect(db_path)
    
    # Load different analysis tables into DataFrames
    funnel_df = pd.read_sql_query("SELECT * FROM funnel_analysis", conn)
    anomalies_df = pd.read_sql_query("SELECT * FROM anomalies", conn)
    payment_method_df = pd.read_sql_query("SELECT * FROM payment_method_performance", conn)
    device_df = pd.read_sql_query("SELECT * FROM device_performance", conn)
    country_df = pd.read_sql_query("SELECT * FROM country_performance", conn)
    hourly_volume_df = pd.read_sql_query("SELECT * FROM hourly_volume", conn)
    hourly_success_df = pd.read_sql_query("SELECT * FROM hourly_success", conn)
    daily_funnel_df = pd.read_sql_query("SELECT * FROM daily_funnel", conn)
    
    # Load A/B test results if available
    try:
        ab_test_df = pd.read_sql_query("SELECT * FROM ab_test_results", conn)
        ab_funnel_df = pd.read_sql_query("SELECT * FROM ab_test_funnel_analysis", conn)
    except:
        ab_test_df = None
        ab_funnel_df = None
    
    conn.close()
    
    # Convert timestamp to datetime
    if 'timestamp' in hourly_volume_df.columns:
        hourly_volume_df['timestamp'] = pd.to_datetime(hourly_volume_df['timestamp'])
    if 'timestamp' in hourly_success_df.columns:
        hourly_success_df['timestamp'] = pd.to_datetime(hourly_success_df['timestamp'])
    if 'date' in daily_funnel_df.columns:
        daily_funnel_df['date'] = pd.to_datetime(daily_funnel_df['date'])
    
    return {
        'funnel': funnel_df,
        'anomalies': anomalies_df,
        'payment_method': payment_method_df,
        'device': device_df,
        'country': country_df,
        'hourly_volume': hourly_volume_df,
        'hourly_success': hourly_success_df,
        'daily_funnel': daily_funnel_df,
        'ab_test': ab_test_df,
        'ab_funnel': ab_funnel_df
    }

# Function to load raw transaction data for custom filtering
@st.cache_data(ttl=3600)
def load_transaction_data(db_path='payment_data.db'):
    """Load raw transaction data from SQLite database"""
    conn = sqlite3.connect(db_path)
    
    # Load different analysis tables into DataFrames
    transactions_df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    
    # Convert timestamp to datetime
    if 'timestamp' in transactions_df.columns:
        transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
    
    return transactions_df

# Custom function to create metric cards
def metric_card(title, value, delta=None, delta_color="normal"):
    """Create a custom metric card with styling"""
    if delta is not None:
        delta_text = f"{delta:.2f}%" if isinstance(delta, (int, float)) else delta
        if delta_color == "inverse":
            delta_color_class = "danger-text" if float(delta) > 0 else "success-text" if float(delta) < 0 else ""
        else:
            delta_color_class = "success-text" if float(delta) > 0 else "danger-text" if float(delta) < 0 else ""
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:14px; color:gray;">{title}</div>
            <div style="font-size:24px; font-weight:bold;">{value}</div>
            <div class="{delta_color_class}">{delta_text}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:14px; color:gray;">{title}</div>
            <div style="font-size:24px; font-weight:bold;">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# Function to create the conversion funnel visualization
def plot_conversion_funnel(funnel_df):
    """Create a conversion funnel visualization"""
    fig = go.Figure()
    
    # Add bars for each stage
    fig.add_trace(go.Funnel(
        name='Count',
        y=funnel_df['stage'],
        x=funnel_df['count'],
        textposition="inside",
        textinfo="value+percent initial",
        opacity=0.65,
        marker={"color": ["#0099C6", "#30A0D1", "#4FA7DC", "#6EAFE8", "#8DB6F3"]},
        connector={"line": {"color": "royalblue", "width": 1}}
    ))
    
    # Add drop-off rates
    drop_off_annotations = []
    for i in range(len(funnel_df) - 1):
        drop_off_annotations.append(
            dict(
                x=funnel_df['count'].iloc[i] - (funnel_df['count'].iloc[i] - funnel_df['count'].iloc[i+1])/2,
                y=funnel_df['stage'].iloc[i],
                text=f"-{funnel_df['drop_off_rate'].iloc[i]:.1f}%",
                showarrow=False,
                font=dict(size=12, color="red"),
                xanchor="center",
                yanchor="bottom"
            )
        )
    
    fig.update_layout(
        title="Conversion Funnel",
        font=dict(size=14),
        height=500,
        annotations=drop_off_annotations
    )
    
    return fig

# Function to create the A/B test comparison chart
def plot_ab_test_comparison(ab_test_df, test_name):
    """Create a visualization for A/B test results comparison"""
    # Filter data for the selected test
    test_data = ab_test_df[ab_test_df['test_name'] == test_name]
    
    # Create a bar chart
    fig = go.Figure()
    
    # Add bars for each variant
    colors = ['#64B5F6', '#42A5F5']  # Light blue and slightly darker blue
    for i, (idx, row) in enumerate(test_data.iterrows()):
        fig.add_trace(go.Bar(
            x=[row['variant']],
            y=[row['conversion_rate']],
            text=[f"{row['conversion_rate']:.2f}%"],
            textposition='auto',
            name=row['variant'],
            marker_color=colors[i % len(colors)]
        ))
    
    # Add significance indicator if available
    if test_data['p_value'].notnull().any():
        p_value = test_data['p_value'].dropna().iloc[0]
        significant = test_data['significant'].dropna().iloc[0]
        relative_improvement = test_data['relative_improvement'].dropna().iloc[0]
        
        title_text = f"{test_name} - Conversion Rate Comparison<br>"
        if significant:
            title_text += f"<span style='color:green'>Statistically Significant (p={p_value:.4f})</span><br>"
            title_text += f"<span style='color:green'>Improvement: {relative_improvement:.2f}%</span>"
        else:
            title_text += f"<span style='color:red'>Not Statistically Significant (p={p_value:.4f})</span>"
    else:
        title_text = f"{test_name} - Conversion Rate Comparison"
    
    fig.update_layout(
        title=title_text,
        font=dict(size=14),
        height=400,
        xaxis_title="Variant",
        yaxis_title="Conversion Rate (%)",
        showlegend=False,
        uniformtext_minsize=10,
        uniformtext_mode='hide'
    )
    
    return fig

# Function to plot A/B test funnel comparison
def plot_ab_test_funnel_comparison(ab_funnel_df, test_name):
    """Create a visualization comparing funnels for each variant in an A/B test"""
    # Filter data for the selected test
    test_data = ab_funnel_df[ab_funnel_df['test_name'] == test_name]
    
    # Create subplots
    variants = test_data['variant'].unique()
    
    # Create a subplot with 1 row and len(variants) columns
    fig = make_subplots(rows=1, cols=len(variants), 
                        shared_yaxes=True,
                        subplot_titles=[v for v in variants])
    
    colors = ["#0099C6", "#30A0D1", "#4FA7DC", "#6EAFE8", "#8DB6F3"]
    
    # Add funnel for each variant
    for i, variant in enumerate(variants):
        variant_data = test_data[test_data['variant'] == variant]
        
        # Sort by funnel stage order
        stage_order = ['initiated', 'details_entered', 'otp_verification', 'processing', 'completed']
        variant_data = variant_data.set_index('funnel_stage').loc[stage_order].reset_index()
        
        fig.add_trace(
            go.Funnel(
                name=variant,
                y=variant_data['funnel_stage'],
                x=variant_data['count'],
                textposition="inside",
                textinfo="value+percent initial",
                opacity=0.65,
                marker={"color": colors},
                showlegend=False
            ),
            row=1, col=i+1
        )
        
        # Add drop-off annotations
        for j in range(len(variant_data) - 1):
            if not pd.isna(variant_data['drop_off_rate'].iloc[j]):
                fig.add_annotation(
                    x=variant_data['count'].iloc[j] - (variant_data['count'].iloc[j] - variant_data['count'].iloc[j+1])/2,
                    y=variant_data['funnel_stage'].iloc[j],
                    text=f"-{variant_data['drop_off_rate'].iloc[j]:.1f}%",
                    showarrow=False,
                    font=dict(size=10, color="red"),
                    xanchor="center",
                    yanchor="bottom",
                    row=1, col=i+1
                )
    
    fig.update_layout(
        title_text=f"{test_name} - Funnel Comparison",
        height=500
    )
    
    return fig

# Function to plot anomaly detection
def plot_anomalies(hourly_volume_df, anomalies_df):
    """Create a time series visualization with anomalies highlighted"""
    # if 'timestamp' not in anomalies_df.columns or 'timestamp' not in hourly_volume_df.columns:
    #     return go.Figure().update_layout(title="No time data available for anomaly visualization")
    
    if 'timestamp' not in anomalies_df.columns or 'timestamp' not in hourly_volume_df.columns:
        return go.Figure().update_layout(title="No time data available for anomaly visualization")
    
    # Ensure timestamps are datetime objects
    anomalies_df = anomalies_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(anomalies_df['timestamp']):
        try:
            anomalies_df['timestamp'] = pd.to_datetime(anomalies_df['timestamp'])
        except Exception as e:
            return go.Figure().update_layout(title=f"Error converting timestamps: {str(e)}")
    
    # Group anomalies by hour
    anomalies_df['hour'] = anomalies_df['timestamp'].dt.floor('h')
    anomaly_hours = anomalies_df.groupby('hour')['transaction_id'].count().reset_index()
    anomaly_hours.columns = ['timestamp', 'anomaly_count']
    
    # Merge with hourly volume
    merged_df = pd.merge(hourly_volume_df, anomaly_hours, on='timestamp', how='left')
    merged_df['anomaly_count'] = merged_df['anomaly_count'].fillna(0)
    merged_df['anomaly_percentage'] = (merged_df['anomaly_count'] / merged_df['transaction_count'] * 100).fillna(0)
    
    # Create time series plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add transaction volume line
    fig.add_trace(
        go.Scatter(
            x=merged_df['timestamp'],
            y=merged_df['transaction_count'],
            name="Transaction Volume",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    
    # Add anomaly percentage line
    fig.add_trace(
        go.Scatter(
            x=merged_df['timestamp'],
            y=merged_df['anomaly_percentage'],
            name="Anomaly %",
            line=dict(color='red', width=1.5, dash='dot')
        ),
        secondary_y=True
    )
    
    # Highlight hours with anomalies above threshold
    anomaly_threshold = 5  # Percentage threshold for highlighting
    highlight_hours = merged_df[merged_df['anomaly_percentage'] > anomaly_threshold]
    
    if not highlight_hours.empty:
        fig.add_trace(
            go.Scatter(
                x=highlight_hours['timestamp'],
                y=highlight_hours['transaction_count'],
                mode='markers',
                name='Anomaly Detected',
                marker=dict(color='red', size=10, symbol='circle')
            ),
            secondary_y=False
        )
    
    # Update layout
    fig.update_layout(
        title="Transaction Volume with Anomaly Detection",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
    fig.update_yaxes(title_text="Anomaly Percentage (%)", secondary_y=True)
    
    return fig

# Function to plot segment performance
def plot_segment_performance(segment_df, column_name, metric_name="Success Rate"):
    """Create a bar chart for segment performance"""
    fig = px.bar(
        segment_df,
        x=column_name,
        y='success_rate',
        title=f"{column_name.replace('_', ' ').title()} {metric_name}",
        labels={'success_rate': metric_name, column_name: column_name.replace('_', ' ').title()},
        color='success_rate',
        color_continuous_scale=['#EF5350', '#FFEE58', '#66BB6A'],
        text=segment_df['success_rate'].apply(lambda x: f"{x:.1f}%")
    )
    
    fig.update_layout(
        height=400,
        coloraxis_showscale=False,
        uniformtext_minsize=10,
        uniformtext_mode='hide'
    )
    
    return fig

# -------------- DASHBOARD LAYOUT --------------

def main():
    # Load the data
    data = load_data()
    
    # Check if we should attempt to load transaction data
    load_transactions = st.sidebar.checkbox("Enable Custom Filtering (Loads Raw Data)", value=False)
    if load_transactions:
        try:
            transactions_df = load_transaction_data()
            has_transactions = True
        except:
            st.sidebar.error("Failed to load transaction data. Using pre-computed analysis only.")
            has_transactions = False
    else:
        has_transactions = False
    
    # Dashboard header
    st.markdown('<div class="main-header">Payment Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Conversion Funnel", "A/B Test Results", "Anomaly Detection", "Segment Analysis"]
    )
    
    # Date filter in sidebar (if transaction data is available)
    if has_transactions:
        st.sidebar.markdown("### Data Filters")
        
        # Date range selection
        min_date = transactions_df['timestamp'].min().date()
        max_date = transactions_df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = transactions_df[
                (transactions_df['timestamp'].dt.date >= start_date) &
                (transactions_df['timestamp'].dt.date <= end_date)
            ]
            
            # Add additional filters
            payment_methods = st.sidebar.multiselect(
                "Payment Methods",
                options=sorted(transactions_df['payment_method'].unique()),
                default=[]
            )
            
            if payment_methods:
                filtered_df = filtered_df[filtered_df['payment_method'].isin(payment_methods)]
            
            device_types = st.sidebar.multiselect(
                "Device Types",
                options=sorted(transactions_df['device_type'].unique()),
                default=[]
            )
            
            if device_types:
                filtered_df = filtered_df[filtered_df['device_type'].isin(device_types)]
            
            # Show filter summary
            st.sidebar.markdown(f"**Filtered Data:** {len(filtered_df)} rows")
        else:
            filtered_df = transactions_df
            st.sidebar.warning("Please select a date range")
    
    # OVERVIEW PAGE
    if page == "Overview":
        # Top KPIs
        st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
        
        # Calculate KPIs from funnel data
        if not data['funnel'].empty:
            initiated = data['funnel'][data['funnel']['stage'] == 'initiated']['count'].values[0]
            completed = data['funnel'][data['funnel']['stage'] == 'completed']['count'].values[0]
            conversion_rate = (completed / initiated * 100) if initiated > 0 else 0
            
            # Get success rate from payment method data
            if not data['payment_method'].empty:
                avg_success_rate = data['payment_method']['success_rate'].mean()
            else:
                avg_success_rate = 0
            
            # Row with KPI metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                metric_card("Total Transactions", f"{initiated:,}")
            with col2:
                metric_card("Completed Transactions", f"{completed:,}")
            with col3:
                metric_card("Conversion Rate", f"{conversion_rate:.2f}%")
            with col4:
                metric_card("Success Rate", f"{avg_success_rate:.2f}%")
        
        # Transaction Volume Time Series
        st.markdown('<div class="section-header">Transaction Volume Over Time</div>', unsafe_allow_html=True)
        
        if not data['hourly_volume'].empty:
            # Create time series chart
            fig = px.line(
                data['hourly_volume'],
                x='timestamp',
                y='transaction_count',
                title="Hourly Transaction Volume",
                labels={'transaction_count': 'Transaction Count', 'timestamp': 'Time'}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hourly volume data available")
        
        # Success Rate Time Series
        if not data['hourly_success'].empty:
            # Create time series chart for success rate
            fig = px.line(
                data['hourly_success'],
                x='timestamp',
                y='success_rate',
                title="Hourly Success Rate",
                labels={'success_rate': 'Success Rate (%)', 'timestamp': 'Time'}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Payment Methods and Device Types
        st.markdown('<div class="section-header">Segment Performance</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not data['payment_method'].empty:
                fig = plot_segment_performance(data['payment_method'], 'payment_method')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if not data['device'].empty:
                fig = plot_segment_performance(data['device'], 'device_type')
                st.plotly_chart(fig, use_container_width=True)
    
    # CONVERSION FUNNEL PAGE
    elif page == "Conversion Funnel":
        st.markdown('<div class="section-header">Conversion Funnel Analysis</div>', unsafe_allow_html=True)
        
        if not data['funnel'].empty:
            # Create the funnel visualization
            fig = plot_conversion_funnel(data['funnel'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Funnel stage metrics
            st.markdown('<div class="subsection-header">Funnel Stage Metrics</div>', unsafe_allow_html=True)
            
            # Create a table with funnel metrics
            funnel_metrics = data['funnel'].copy()
            
            # Add continuation rate (inverse of drop-off)
            funnel_metrics['continuation_rate'] = 100 - funnel_metrics['drop_off_rate']
            
            # Format the table
            funnel_table = funnel_metrics[['stage', 'count', 'drop_off_rate', 'continuation_rate']]
            funnel_table.columns = ['Stage', 'Count', 'Drop-off Rate (%)', 'Continuation Rate (%)']
            
            # Display the table
            st.dataframe(funnel_table.style.format({
                'Count': '{:,}',
                'Drop-off Rate (%)': '{:.2f}%',
                'Continuation Rate (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # Daily Funnel Progression
            st.markdown('<div class="subsection-header">Daily Funnel Progression</div>', unsafe_allow_html=True)
            
            if not data['daily_funnel'].empty:
                # Pivot the data to get stages as columns
                daily_pivot = data['daily_funnel'].pivot(index='date', columns='funnel_stage', values='transaction_id').reset_index()
                
                # Fill NaN with 0
                daily_pivot = daily_pivot.fillna(0)
                
                # Create a line chart with multiple lines for each stage
                fig = px.line(
                    daily_pivot, 
                    x='date', 
                    y=['initiated', 'details_entered', 'otp_verification', 'processing', 'completed'],
                    title="Daily Funnel Progression",
                    labels={'date': 'Date', 'value': 'Transaction Count', 'variable': 'Funnel Stage'}
                )
                
                fig.update_layout(height=500, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No daily funnel data available")
        else:
            st.info("No funnel data available")
    
    # A/B TEST RESULTS PAGE
    elif page == "A/B Test Results":
        st.markdown('<div class="section-header">A/B Test Results</div>', unsafe_allow_html=True)
        
        if data['ab_test'] is not None and not data['ab_test'].empty:
            # Get unique test names
            test_names = data['ab_test']['test_name'].unique()
            
            # Let user select a test to view
            selected_test = st.selectbox("Select A/B Test", test_names)
            
            # Show test results
            if selected_test:
                # Test overview
                st.markdown('<div class="subsection-header">Test Overview</div>', unsafe_allow_html=True)
                
                # Filter data for the selected test
                test_data = data['ab_test'][data['ab_test']['test_name'] == selected_test]
                
                # Display test metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create bar chart for conversion rates
                    fig = plot_ab_test_comparison(data['ab_test'], selected_test)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Create a details table
                    metrics_table = test_data[['variant', 'sample_size', 'conversion_rate']].copy()
                    metrics_table.columns = ['Variant', 'Sample Size', 'Conversion Rate']
                    
                    # Add statistical significance info
                    if test_data['p_value'].notnull().any():
                        p_value = test_data['p_value'].dropna().iloc[0]
                        significant = test_data['significant'].dropna().iloc[0]
                        relative_improvement = test_data['relative_improvement'].dropna().iloc[0]
                        
                        significance_text = "Yes ✓" if significant else "No ✗"
                        
                        st.markdown(f"""
                        ### Statistical Significance
                        - **Significant**: {significance_text}
                        - **P-value**: {p_value:.4f}
                        - **Relative Improvement**: {relative_improvement:.2f}%
                        """)
                    
                    # Display the table
                    st.dataframe(metrics_table.style.format({
                        'Sample Size': '{:,}',
                        'Conversion Rate': '{:.2f}%'
                    }), use_container_width=True)
                
                # Show funnel comparison
                if data['ab_funnel'] is not None and not data['ab_funnel'].empty:
                    st.markdown('<div class="subsection-header">Funnel Comparison</div>', unsafe_allow_html=True)
                    
                    # Create funnel comparison visualization
                    fig = plot_ab_test_funnel_comparison(data['ab_funnel'], selected_test)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create a table with drop-off rates by stage
                    funnel_data = data['ab_funnel'][data['ab_funnel']['test_name'] == selected_test].copy()
                    
                    # Pivot to get variants as columns and funnel stages as rows
                    drop_off_pivot = funnel_data.pivot_table(
                        index='funnel_stage', 
                        columns='variant', 
                        values='drop_off_rate'
                    ).reset_index()
                    
                    # Order stages correctly
                    stage_order = ['initiated', 'details_entered', 'otp_verification', 'processing', 'completed']
                    
                    # Check which stages actually exist in the data
                    available_stages = drop_off_pivot.index.tolist()
                    # Only use stages that exist in both lists
                    valid_stages = [stage for stage in stage_order if stage in available_stages]
                                        
                    if valid_stages:
                        drop_off_pivot = drop_off_pivot.loc[valid_stages].reset_index()
                    else:
                        # If no stages match, just use the data as is
                        drop_off_pivot = drop_off_pivot.reset_index()
                    
                    # drop_off_pivot = drop_off_pivot.set_index('funnel_stage').loc[stage_order].reset_index()
                    
                    # Display the table
                    st.markdown("#### Drop-off Rates by Stage (%)")
                    st.dataframe(drop_off_pivot.style.format({col: '{:.2f}%' for col in drop_off_pivot.columns if col != 'funnel_stage'}), 
                                use_container_width=True)
            else:
                st.info("Please select an A/B test to view results")
        else:
            st.info("No A/B test data available")
    
    # ANOMALY DETECTION PAGE
    elif page == "Anomaly Detection":
        st.markdown('<div class="section-header">Anomaly Detection</div>', unsafe_allow_html=True)
        
        if not data['anomalies'].empty:
            # Anomaly summary metrics
            total_anomalies = len(data['anomalies'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Total anomalies detected:** {total_anomalies:,}")
            
            with col2:
                if 'amount' in data['anomalies'].columns:
                    avg_anomaly_amount = data['anomalies']['amount'].mean()
                    st.markdown(f"**Average anomaly amount:** ${avg_anomaly_amount:.2f}")
            
            # Time series chart with anomalies
            if not data['hourly_volume'].empty:
                fig = plot_anomalies(data['hourly_volume'], data['anomalies'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly details table
            st.markdown('<div class="subsection-header">Anomaly Details</div>', unsafe_allow_html=True)
            
            # Select columns to display
            display_cols = ['transaction_id', 'timestamp', 'amount', 'payment_method', 'device_type', 'country']
            display_cols = [col for col in display_cols if col in data['anomalies'].columns]
            
            st.dataframe(data['anomalies'][display_cols].sort_values('timestamp', ascending=False), use_container_width=True)
        else:
            st.info("No anomaly data available")
                            
    # SEGMENT ANALYSIS PAGE
    elif page == "Segment Analysis":
        st.markdown('<div class="section-header">Segment Analysis</div>', unsafe_allow_html=True)
        
        # Tabs for different segment types
        segment_tabs = st.tabs(["Payment Methods", "Device Types", "Countries"])
        
        # Payment Methods Tab
        with segment_tabs[0]:
            if not data['payment_method'].empty:
                # Payment method performance chart
                fig = plot_segment_performance(data['payment_method'], 'payment_method', "Success Rate")
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional payment method metrics
                st.markdown('<div class="subsection-header">Payment Method Details</div>', unsafe_allow_html=True)
                
                # Create a detailed table
                payment_details = data['payment_method'].copy()
                
                # Check if required columns exist
                required_columns = ['success_count', 'failure_count']
                missing_columns = [col for col in required_columns if col not in payment_details.columns]
                
                # payment_details['volume'] = payment_details['success_count'] + payment_details['failure_count']
                # payment_details['volume_percentage'] = payment_details['volume'] / payment_details['volume'].sum() * 100
                
                # # Calculate metrics
                # payment_details = payment_details.sort_values('volume', ascending=False)
                
                # # Display the table
                # st.dataframe(payment_details[[
                #     'payment_method', 'volume', 'volume_percentage', 'success_count', 
                #     'failure_count', 'success_rate'
                # ]].rename(columns={
                #     'payment_method': 'Payment Method',
                #     'volume': 'Transaction Volume',
                #     'volume_percentage': 'Volume %',
                #     'success_count': 'Successful',
                #     'failure_count': 'Failed',
                #     'success_rate': 'Success Rate'
                # }).style.format({
                #     'Transaction Volume': '{:,}',
                #     'Volume %': '{:.2f}%',
                #     'Successful': '{:,}',
                #     'Failed': '{:,}',
                #     'Success Rate': '{:.2f}%'
                # }), use_container_width=True)
                
                
                
                
                
                if missing_columns:
                    st.error(f"Missing columns in payment data: {', '.join(missing_columns)}")
                    st.info(f"Available columns: {', '.join(payment_details.columns)}")
                    
                    # Create a simplified view with available data
                    st.markdown("#### Payment Method Performance")
                    
                    # Use only available columns
                    available_columns = ['payment_method']
                    if 'success_rate' in payment_details.columns:
                        available_columns.append('success_rate')
                    
                    # If we have transaction_count, use it
                    if 'transaction_count' in payment_details.columns:
                        available_columns.append('transaction_count')
                        payment_details = payment_details.sort_values('transaction_count', ascending=False)
                    else:
                        # Just sort alphabetically if no counts available
                        payment_details = payment_details.sort_values('payment_method')
                    
                    # Create column mapping for available columns
                    column_mapping = {
                        'payment_method': 'Payment Method',
                        'success_rate': 'Success Rate',
                        'transaction_count': 'Transaction Count'
                    }
                    
                    # Filter to only available columns
                    display_mapping = {k: column_mapping[k] for k in available_columns if k in column_mapping}
                    
                    # Format based on available columns
                    format_dict = {}
                    if 'success_rate' in available_columns:
                        format_dict['Success Rate'] = '{:.2f}%'
                    if 'transaction_count' in available_columns:
                        format_dict['Transaction Count'] = '{:,}'
                    
                    # Display what we have
                    st.dataframe(payment_details[available_columns].rename(columns=display_mapping)
                                .style.format(format_dict), use_container_width=True)
                else:
                    # Original code path when all columns exist
                    payment_details['volume'] = payment_details['success_count'] + payment_details['failure_count']
                    payment_details['volume_percentage'] = payment_details['volume'] / payment_details['volume'].sum() * 100
                    
                    # Calculate metrics
                    payment_details = payment_details.sort_values('volume', ascending=False)
                    
                    # Display the table
                    st.dataframe(payment_details[[
                        'payment_method', 'volume', 'volume_percentage', 'success_count', 
                        'failure_count', 'success_rate'
                    ]].rename(columns={
                        'payment_method': 'Payment Method',
                        'volume': 'Transaction Volume',
                        'volume_percentage': 'Volume %',
                        'success_count': 'Successful',
                        'failure_count': 'Failed',
                        'success_rate': 'Success Rate'
                    }).style.format({
                        'Transaction Volume': '{:,}',
                        'Volume %': '{:.2f}%',
                        'Successful': '{:,}',
                        'Failed': '{:,}',
                        'Success Rate': '{:.2f}%'
                    }), use_container_width=True)
                
                import random
                
                # For testing/prototyping only
                if 'success_count' not in payment_details.columns:
                    payment_details['success_count'] = [random.randint(800, 1000) for _ in range(len(payment_details))]
                    payment_details['failure_count'] = [random.randint(50, 200) for _ in range(len(payment_details))]
                    st.info("Using synthetic data for visualization purposes")
                
                # if missing_columns:
                #     st.error(f"Missing columns in payment data: {', '.join(missing_columns)}")
                #     # Display the columns that are available
                #     st.info(f"Available columns: {', '.join(payment_details.columns)}")
                # else:
                #     payment_details['volume'] = payment_details['success_count'] + payment_details['failure_count']
                #     payment_details['volume_percentage'] = payment_details['volume'] / payment_details['volume'].sum() * 100
                    
                #     # Calculate metrics
                #     payment_details = payment_details.sort_values('volume', ascending=False)
                    
                #     # Display the table
                #     st.dataframe(payment_details[[
                #         'payment_method', 'volume', 'volume_percentage', 'success_count', 
                #         'failure_count', 'success_rate'
                #     ]].rename(columns={
                #         'payment_method': 'Payment Method',
                #         'volume': 'Transaction Volume',
                #         'volume_percentage': 'Volume %',
                #         'success_count': 'Successful',
                #         'failure_count': 'Failed',
                #         'success_rate': 'Success Rate'
                #     }).style.format({
                #         'Transaction Volume': '{:,}',
                #         'Volume %': '{:.2f}%',
                #         'Successful': '{:,}',
                #         'Failed': '{:,}',
                #         'Success Rate': '{:.2f}%'
                #     }), use_container_width=True)
                            
                
            else:
                st.info("No payment method data available")
        
        # Device Types Tab
        with segment_tabs[1]:
            if not data['device'].empty:
                # Device type performance chart
                fig = plot_segment_performance(data['device'], 'device_type', "Success Rate")
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional device metrics
                st.markdown('<div class="subsection-header">Device Type Details</div>', unsafe_allow_html=True)
                
                # Create a detailed table
                device_details = data['device'].copy()
                
                # Check if required columns exist
                required_columns = ['success_count', 'failure_count']
                missing_columns = [col for col in required_columns if col not in device_details.columns]

                if missing_columns:
                    st.error(f"Missing columns in device data: {', '.join(missing_columns)}")
                    # Display the columns that are available
                    st.info(f"Available columns: {', '.join(device_details.columns)}")
                else:
                    device_details['volume'] = device_details['success_count'] + device_details['failure_count']
                    device_details['volume_percentage'] = device_details['volume'] / device_details['volume'].sum() * 100
                    
                    # Calculate metrics
                    device_details = device_details.sort_values('volume', ascending=False)
                    
                    # Display the table
                    st.dataframe(device_details[[
                        'device_type', 'volume', 'volume_percentage', 'success_count', 
                        'failure_count', 'success_rate'
                    ]].rename(columns={
                        'device_type': 'Device Type',
                        'volume': 'Transaction Volume',
                        'volume_percentage': 'Volume %',
                        'success_count': 'Successful',
                        'failure_count': 'Failed',
                        'success_rate': 'Success Rate'
                    }).style.format({
                        'Transaction Volume': '{:,}',
                        'Volume %': '{:.2f}%',
                        'Successful': '{:,}',
                        'Failed': '{:,}',
                        'Success Rate': '{:.2f}%'
                    }), use_container_width=True)
                
                
                # device_details['volume'] = device_details['success_count'] + device_details['failure_count']
                # device_details['volume_percentage'] = device_details['volume'] / device_details['volume'].sum() * 100
                
                # # Calculate metrics
                # device_details = device_details.sort_values('volume', ascending=False)
                
                # # Display the table
                # st.dataframe(device_details[[
                #     'device_type', 'volume', 'volume_percentage', 'success_count', 
                #     'failure_count', 'success_rate'
                # ]].rename(columns={
                #     'device_type': 'Device Type',
                #     'volume': 'Transaction Volume',
                #     'volume_percentage': 'Volume %',
                #     'success_count': 'Successful',
                #     'failure_count': 'Failed',
                #     'success_rate': 'Success Rate'
                # }).style.format({
                #     'Transaction Volume': '{:,}',
                #     'Volume %': '{:.2f}%',
                #     'Successful': '{:,}',
                #     'Failed': '{:,}',
                #     'Success Rate': '{:.2f}%'
                # }), use_container_width=True)
            else:
                st.info("No device type data available")
        
        # # Countries Tab
        # with segment_tabs[2]:
        #     if not data['country'].empty:
        #         # Country performance chart (top 10)
        #         top_countries = data['country'].sort_values('volume', ascending=False).head(10)
                
        #         fig = plot_segment_performance(top_countries, 'country', "Success Rate")
        #         st.plotly_chart(fig, use_container_width=True)
                
        #         # Additional country metrics
        #         st.markdown('<div class="subsection-header">Country Details</div>', unsafe_allow_html=True)
                
        #         # Create a detailed table
        #         country_details = data['country'].copy()
                
        #         # Check if 'volume' column exists or needs to be calculated
        #         if 'volume' not in country_details.columns:
        #             # Check if we can calculate volume from success and failure counts
        #             if 'success_count' in country_details.columns and 'failure_count' in country_details.columns:
        #                 country_details['volume'] = country_details['success_count'] + country_details['failure_count']
        #             else:
        #                 # Display available columns
        #                 st.error("Missing 'volume' column in country data")
        #                 st.info(f"Available columns: {', '.join(country_details.columns)}")
        #                 # If transaction_count exists, we could use that as a fallback
        #                 if 'transaction_count' in country_details.columns:
        #                     country_details['volume'] = country_details['transaction_count']
        #                 else:
        #                     # Create a dummy column with zeros to prevent errors
        #                     country_details['volume'] = 0
                
                
        #         # country_details['volume'] = country_details['success_count'] + country_details['failure_count']
                
                
                
        #         country_details['volume_percentage'] = country_details['volume'] / country_details['volume'].sum() * 100
                
                
        #         # # Country performance chart (top 10)
        #         # top_countries = data['country'].sort_values('volume', ascending=False).head(10)
                
        #         # fig = plot_segment_performance(top_countries, 'country', "Success Rate")
        #         # st.plotly_chart(fig, use_container_width=True)
                
        #         # # Additional country metrics
        #         # st.markdown('<div class="subsection-header">Country Details</div>', unsafe_allow_html=True)
                
        #         # Calculate metrics
        #         country_details = country_details.sort_values('volume', ascending=False)
                
        #         # Display the table with search functionality
        #         st.dataframe(country_details[[
        #             'country', 'volume', 'volume_percentage', 'success_count', 
        #             'failure_count', 'success_rate'
        #         ]].rename(columns={
        #             'country': 'Country',
        #             'volume': 'Transaction Volume',
        #             'volume_percentage': 'Volume %',
        #             'success_count': 'Successful',
        #             'failure_count': 'Failed',
        #             'success_rate': 'Success Rate'
        #         }).style.format({
        #             'Transaction Volume': '{:,}',
        #             'Volume %': '{:.2f}%',
        #             'Successful': '{:,}',
        #             'Failed': '{:,}',
        #             'Success Rate': '{:.2f}%'
        #         }), use_container_width=True)
                
        #         # Map visualization if geo data is available
        #         if 'country_code' in data['country'].columns:
        #             st.markdown('<div class="subsection-header">Global Success Rate Map</div>', unsafe_allow_html=True)
                    
        #             fig = px.choropleth(
        #                 data['country'],
        #                 locations='country_code',
        #                 color='success_rate',
        #                 hover_name='country',
        #                 color_continuous_scale=['#EF5350', '#FFEE58', '#66BB6A'],
        #                 range_color=[50, 100],
        #                 labels={'success_rate': 'Success Rate (%)'}
        #             )
                    
        #             fig.update_layout(
        #                 height=500,
        #                 geo=dict(
        #                     showframe=False,
        #                     showcoastlines=True,
        #                     projection_type='equirectangular'
        #                 )
        #             )
                    
        #             st.plotly_chart(fig, use_container_width=True)
        #     else:
        #         st.info("No country data available")
        
        
        
        # Countries Tab
        with segment_tabs[2]:
            if not data['country'].empty:
                # Create country_details first and add volume
                country_details = data['country'].copy()
                
                # Check if 'volume' column exists or needs to be calculated
                if 'volume' not in country_details.columns:
                    # Check if we can calculate volume from success and failure counts
                    if 'success_count' in country_details.columns and 'failure_count' in country_details.columns:
                        country_details['volume'] = country_details['success_count'] + country_details['failure_count']
                    else:
                        # Display available columns
                        st.error("Missing 'volume' column in country data")
                        st.info(f"Available columns: {', '.join(country_details.columns)}")
                        # If transaction_count exists, we could use that as a fallback
                        if 'transaction_count' in country_details.columns:
                            country_details['volume'] = country_details['transaction_count']
                        else:
                            # Create a dummy column with zeros to prevent errors
                            country_details['volume'] = 0
                
                # Now that we have volume, get top countries
                top_countries = country_details.sort_values('volume', ascending=False).head(10)
                
                # Continue with visualization
                fig = plot_segment_performance(top_countries, 'country', "Success Rate")
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional country metrics
                st.markdown('<div class="subsection-header">Country Details</div>', unsafe_allow_html=True)
                
                # Now calculate other metrics
                country_details['volume_percentage'] = country_details['volume'] / country_details['volume'].sum() * 100
                country_details = country_details.sort_values('volume', ascending=False)
                
                # Display the table
                # Only include columns that exist
                available_columns = ['country', 'volume', 'volume_percentage']
                for col in ['success_count', 'failure_count', 'success_rate']:
                    if col in country_details.columns:
                        available_columns.append(col)
                
                # Create a mapping for only the columns we have
                column_mapping = {
                    'country': 'Country',
                    'volume': 'Transaction Volume',
                    'volume_percentage': 'Volume %',
                    'success_count': 'Successful',
                    'failure_count': 'Failed',
                    'success_rate': 'Success Rate'
                }
                
                # Keep only mappings for columns that exist
                display_mapping = {k: column_mapping[k] for k in available_columns if k in column_mapping}
                
                # Create format dict only for columns that exist
                format_dict = {}
                if 'volume' in available_columns:
                    format_dict['Transaction Volume'] = '{:,}'
                if 'volume_percentage' in available_columns:
                    format_dict['Volume %'] = '{:.2f}%'
                if 'success_count' in available_columns:
                    format_dict['Successful'] = '{:,}'
                if 'failure_count' in available_columns:
                    format_dict['Failed'] = '{:,}'
                if 'success_rate' in available_columns:
                    format_dict['Success Rate'] = '{:.2f}%'
                
                # Now display with only the columns we have
                st.dataframe(country_details[available_columns].rename(columns=display_mapping)
                            .style.format(format_dict), use_container_width=True)
                
                # Map visualization if geo data is available
                if 'country_code' in data['country'].columns:
                    st.markdown('<div class="subsection-header">Global Success Rate Map</div>', unsafe_allow_html=True)
                    
                    # Make sure success_rate exists before using it
                    if 'success_rate' in data['country'].columns:
                        fig = px.choropleth(
                            data['country'],
                            locations='country_code',
                            color='success_rate',
                            hover_name='country',
                            color_continuous_scale=['#EF5350', '#FFEE58', '#66BB6A'],
                            range_color=[50, 100],
                            labels={'success_rate': 'Success Rate (%)'}
                        )
                        
                        fig.update_layout(
                            height=500,
                            geo=dict(
                                showframe=False,
                                showcoastlines=True,
                                projection_type='equirectangular'
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No success rate data available for map visualization")
            else:
                st.info("No country data available")

    # Add custom filtering functionality if transaction data is loaded
    if has_transactions and page == "Segment Analysis":
        st.markdown('<div class="section-header">Custom Segment Analysis</div>', unsafe_allow_html=True)
        
        st.write("Create your own segment analysis by selecting dimensions and metrics below.")
        
        # Select dimensions for analysis
        dimension_options = [col for col in transactions_df.columns if transactions_df[col].dtype == 'object']
        selected_dimension = st.selectbox("Select Dimension", dimension_options)
        
        if selected_dimension:
            # Calculate metrics based on the selected dimension
            segment_counts = filtered_df.groupby(selected_dimension)['transaction_id'].count().reset_index()
            segment_counts.columns = [selected_dimension, 'count']
            
            # Calculate success rate if status column exists
            if 'status' in filtered_df.columns:
                success_counts = filtered_df[filtered_df['status'] == 'completed'].groupby(selected_dimension)['transaction_id'].count().reset_index()
                success_counts.columns = [selected_dimension, 'success_count']
                
                # Merge with total counts
                segment_analysis = pd.merge(segment_counts, success_counts, on=selected_dimension, how='left')
                segment_analysis['success_count'] = segment_analysis['success_count'].fillna(0)
                segment_analysis['success_rate'] = segment_analysis['success_count'] / segment_analysis['count'] * 100
                
                # Sort by count
                segment_analysis = segment_analysis.sort_values('count', ascending=False)
                
                # Display results
                st.markdown(f"<div class='subsection-header'>Analysis by {selected_dimension.replace('_', ' ').title()}</div>", unsafe_allow_html=True)
                
                # Create visualization
                fig = px.bar(
                    segment_analysis.head(10),
                    x=selected_dimension,
                    y='success_rate',
                    title=f"Top 10 {selected_dimension.replace('_', ' ').title()} by Success Rate",
                    labels={'success_rate': 'Success Rate (%)', selected_dimension: selected_dimension.replace('_', ' ').title()},
                    color='success_rate',
                    color_continuous_scale=['#EF5350', '#FFEE58', '#66BB6A'],
                    text=segment_analysis['success_rate'].head(10).apply(lambda x: f"{x:.1f}%")
                )
                
                fig.update_layout(
                    height=400,
                    coloraxis_showscale=False,
                    uniformtext_minsize=10,
                    uniformtext_mode='hide'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the data table
                st.dataframe(segment_analysis[[
                    selected_dimension, 'count', 'success_count', 'success_rate'
                ]].rename(columns={
                    selected_dimension: selected_dimension.replace('_', ' ').title(),
                    'count': 'Transaction Volume',
                    'success_count': 'Successful',
                    'success_rate': 'Success Rate'
                }).style.format({
                    'Transaction Volume': '{:,}',
                    'Successful': '{:,}',
                    'Success Rate': '{:.2f}%'
                }), use_container_width=True)
            else:
                # Display just the counts if status not available
                segment_counts = segment_counts.sort_values('count', ascending=False)
                
                # Create visualization
                fig = px.bar(
                    segment_counts.head(10),
                    x=selected_dimension,
                    y='count',
                    title=f"Top 10 {selected_dimension.replace('_', ' ').title()} by Transaction Volume",
                    labels={'count': 'Transaction Count', selected_dimension: selected_dimension.replace('_', ' ').title()},
                    color='count',
                    color_continuous_scale=['#90CAF9', '#42A5F5', '#1976D2'],
                    text=segment_counts['count'].head(10).apply(lambda x: f"{x:,}")
                )
                
                fig.update_layout(
                    height=400,
                    coloraxis_showscale=False,
                    uniformtext_minsize=10,
                    uniformtext_mode='hide'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the data table
                st.dataframe(segment_counts.rename(columns={
                    selected_dimension: selected_dimension.replace('_', ' ').title(),
                    'count': 'Transaction Volume'
                }).style.format({
                    'Transaction Volume': '{:,}'
                }), use_container_width=True)

    # Add download functionality
    if st.sidebar.checkbox("Enable Data Export"):
        st.sidebar.markdown("### Export Data")
        
        # Export options
        export_format = st.sidebar.selectbox(
            "Export Format",
            ["CSV", "Excel"]
        )
        
        if export_format == "CSV":
            if 'funnel' in data and not data['funnel'].empty:
                funnel_csv = data['funnel'].to_csv(index=False).encode('utf-8')
                st.sidebar.download_button(
                    label="Download Funnel Data",
                    data=funnel_csv,
                    file_name="funnel_data.csv",
                    mime="text/csv"
                )
            
            if has_transactions:
                filtered_csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.sidebar.download_button(
                    label="Download Filtered Transactions",
                    data=filtered_csv,
                    file_name="filtered_transactions.csv",
                    mime="text/csv"
                )
        else:  # Excel
            try:
                import io
                from io import BytesIO
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    for key, df in data.items():
                        if df is not None and not df.empty:
                            sheet_name = key[:31]  # Excel has 31 char limit for sheet names
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    if has_transactions:
                        filtered_df.to_excel(writer, sheet_name="Filtered Transactions", index=False)
                
                excel_data = output.getvalue()
                st.sidebar.download_button(
                    label="Download All Data (Excel)",
                    data=excel_data,
                    file_name="payment_analytics.xlsx",
                    mime="application/vnd.ms-excel"
                )
            except Exception as e:
                st.sidebar.error(f"Excel export failed: {str(e)}")

# About section in sidebar
with st.sidebar.expander("About"):
    st.markdown("""
    **Payment Analytics Dashboard** v1.0
    
    A comprehensive analytics solution for payment transaction data. 
    Features include conversion funnel analysis, anomaly detection, A/B testing, 
    and segment analysis.
    
    Data is loaded from SQLite databases:
    - `analysis_results.db`: Pre-computed analysis
    - `payment_data.db`: Raw transaction data (when enabled)
    
    For support or feature requests, contact the development team.
    """)

# Run the dashboard
if __name__ == "__main__":
    main()
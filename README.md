# Payment Mission Control Center 💸

A comprehensive real-time payment monitoring system that helps detect anomalies, analyze conversion funnels, and optimize payment workflows.

![Dashboard Preview](https://github.com/user-attachments/assets/bf7e1cfc-17dd-4b60-9491-4de5945f5e71)

## 🚀 Overview

The Payment Mission Control Center is a sophisticated dashboard application designed to provide real-time insights into payment system performance. It enables payment operations teams to:

- Monitor transaction volumes and success rates in real-time
- Analyze the conversion funnel to identify drop-off points
- Detect anomalies that might indicate fraud or system issues
- Explore transaction data across different segments (payment methods, devices, countries)
- Generate actionable recommendations to improve conversion rates

## ✨ Features

- **Real-time Dashboard**: Interactive visualization of key payment metrics
- **Conversion Funnel Analysis**: Track user journey from initiation to completion
- **Anomaly Detection**: Automatically identify unusual transaction patterns
- **Segmentation Analysis**: Compare performance across payment methods, devices, and regions
- **Transaction Explorer**: Filter and download transaction data for offline analysis
- **Recommendation Engine**: Get actionable insights to improve payment performance

## 🔧 Technologies

- **Python 3.8+**: Core programming language
- **Streamlit**: Interactive dashboard framework
- **Pandas & NumPy**: Data manipulation and analysis
- **SQLite**: Lightweight database for storing transaction data
- **Plotly**: Interactive data visualization
- **SciPy**: Statistical analysis for anomaly detection
- **Faker**: Generation of realistic test data

## 📋 Project Structure

- `payment_data_generator.py`: Creates synthetic payment transaction data
- `payment_analysis.py`: Analyzes transaction data and generates insights
- `dashboard.py`: Streamlit dashboard for visualizing the analysis
- `payment_data.db`: SQLite database for raw transaction data
- `analysis_results.db`: SQLite database for processed analysis results

## 🔍 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/payment-mission-control.git
   cd payment-mission-control
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies
Create a `requirements.txt` file with the following contents:

```
streamlit>=1.18.0
pandas>=1.4.0
numpy>=1.22.0
faker>=13.0.0
plotly>=5.6.0
scipy>=1.8.0
matplotlib>=3.5.0
seaborn>=0.11.2
```

## 🚦 Running the Project

1. Generate synthetic payment data:
   ```bash
   python payment_data_generator.py
   ```

2. Run the data analysis script:
   ```bash
   python payment_analysis.py
   ```

3. Launch the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

4. Open your browser and navigate to `http://localhost:8501` to view the dashboard

## 📊 Sample Visualizations

The dashboard includes several key visualizations:

- Payment Conversion Funnel
- Transaction Volume Over Time
- Success Rate by Payment Method
- Anomaly Detection Visualization
- Drop-off Analysis by Funnel Stage

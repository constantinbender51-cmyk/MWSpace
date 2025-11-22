import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
from flask import Flask, render_template_string
import warnings
warnings.filterwarnings('ignore')

# Initialize Binance client
client = Client()

def fetch_btc_data():
    """Fetch BTC/USDT 1-minute data from Binance"""
    print("Fetching BTC/USDT 1-minute data from Binance...")
    
    # Define date range
    start_date = "2022-01-01"
    end_date = "2023-09-30"
    
    # Convert dates to milliseconds
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    current_start = start_ts
    
    # Binance API has limits, so we need to fetch in chunks
    while current_start < end_ts:
        current_end = min(current_start + 1000 * 60 * 1000, end_ts)  # 1000 minutes per request
        
        klines = client.get_historical_klines(
            "BTCUSDT",
            Client.KLINE_INTERVAL_1MINUTE,
            current_start,
            current_end
        )
        
        if not klines:
            break
            
        all_data.extend(klines)
        current_start = current_end + 60000  # Move to next minute
        
        print(f"Fetched {len(klines)} records... Total: {len(all_data)}")
    
    # Convert to DataFrame
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    
    df = pd.DataFrame(all_data, columns=columns)
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])
    
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    print(f"Total records fetched: {len(df)}")
    return df

def prepare_features(df, lookback=10):
    """Prepare features for the linear regression model"""
    print("Preparing features...")
    
    # Create lagged features
    for i in range(1, lookback + 1):
        df[f'close_lag_{i}'] = df['close'].shift(i)
        df[f'volume_lag_{i}'] = df['volume'].shift(i)
    
    # Create rolling statistics
    df['close_rolling_mean_5'] = df['close'].rolling(window=5).mean()
    df['close_rolling_std_5'] = df['close'].rolling(window=5).std()
    df['volume_rolling_mean_5'] = df['volume'].rolling(window=5).mean()
    
    # Price changes
    df['price_change'] = df['close'].pct_change()
    
    # Remove rows with NaN values
    df = df.dropna()
    
    return df

def train_linear_regression(df):
    """Train linear regression model and make predictions"""
    print("Training linear regression model...")
    
    # Define features and target
    feature_columns = [col for col in df.columns if col not in ['close', 'price_change']]
    X = df[feature_columns]
    y = df['close']
    
    # Split data (40% test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, shuffle=False, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_columns': feature_columns
    }

def create_plot(results):
    """Create comparison plot of predictions vs actual prices"""
    print("Creating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Training set results
    ax1.plot(results['y_train'].values[:1000], label='Actual', alpha=0.7, linewidth=1)
    ax1.plot(results['y_pred_train'][:1000], label='Predicted', alpha=0.7, linewidth=1)
    ax1.set_title('Linear Regression - Training Set (First 1000 samples)')
    ax1.set_ylabel('BTC Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test set results
    test_samples = min(1000, len(results['y_test']))
    ax2.plot(results['y_test'].values[:test_samples], label='Actual', alpha=0.7, linewidth=1)
    ax2.plot(results['y_pred_test'][:test_samples], label='Predicted', alpha=0.7, linewidth=1)
    ax2.set_title('Linear Regression - Test Set (40% of data)')
    ax2.set_ylabel('BTC Price (USDT)')
    ax2.set_xlabel('Time Index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert plot to base64 for HTML display
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

# HTML template for the web server
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BTC Price Prediction - Linear Regression</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 30px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .plot {
            text-align: center;
            margin: 30px 0;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .info {
            background: #e7f3ff;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>BTC Price Prediction using Linear Regression</h1>
        
        <div class="info">
            <strong>Data Period:</strong> January 2022 - September 2023<br>
            <strong>Timeframe:</strong> 1-minute<br>
            <strong>Test Set Size:</strong> 40% of total data<br>
            <strong>Model:</strong> Linear Regression with lagged features
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <h3>Training RMSE</h3>
                <div class="metric-value">{{ train_rmse }}</div>
            </div>
            <div class="metric-card">
                <h3>Test RMSE</h3>
                <div class="metric-value">{{ test_rmse }}</div>
            </div>
            <div class="metric-card">
                <h3>Training R² Score</h3>
                <div class="metric-value">{{ train_r2 }}</div>
            </div>
            <div class="metric-card">
                <h3>Test R² Score</h3>
                <div class="metric-value">{{ test_r2 }}</div>
            </div>
        </div>
        
        <div class="plot">
            <h2>Predictions vs Actual Prices</h2>
            <img src="data:image/png;base64,{{ plot_url }}" alt="BTC Price Prediction Results">
        </div>
        
        <div class="info">
            <strong>Note:</strong> Linear regression on financial time series data, especially at 1-minute intervals, 
            may not capture complex market dynamics. This is for educational purposes.
        </div>
    </div>
</body>
</html>
"""

def main():
    """Main function to run the entire pipeline"""
    try:
        # Fetch data
        df = fetch_btc_data()
        
        if len(df) == 0:
            print("No data fetched. Please check your internet connection and try again.")
            return
        
        # Prepare features
        df_processed = prepare_features(df)
        
        # Train model
        results = train_linear_regression(df_processed)
        
        # Create plot
        plot_url = create_plot(results)
        
        # Start web server
        app = Flask(__name__)
        
        @app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE,
                train_rmse=f"{results['train_rmse']:.2f}",
                test_rmse=f"{results['test_rmse']:.2f}",
                train_r2=f"{results['train_r2']:.4f}",
                test_r2=f"{results['test_r2']:.4f}",
                plot_url=plot_url
            )
        
        print("\nWeb server starting on http://0.0.0.0:8080")
        print("Press Ctrl+C to stop the server")
        
        app.run(host='0.0.0.0', port=8080, debug=False)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please make sure you have installed all required packages:")
        print("pip install pandas numpy scikit-learn matplotlib flask python-binance")

if __name__ == "__main__":
    main()

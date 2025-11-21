import pandas as pd
import numpy as np
import requests
from binance.client import Client
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from flask import Flask, send_file, jsonify
import io
import datetime
from threading import Thread
import time
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from typing import Dict, List, Tuple
import joblib

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
latest_data = None
latest_covariance = None
latest_predictions = None
models = {}
scalers = {}

# Binance API configuration
BINANCE_API_KEY = 'your_api_key_here'
BINANCE_API_SECRET = 'your_api_secret_here'

def get_binance_client():
    """Initialize Binance client"""
    try:
        return Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    except:
        return Client()

def get_most_liquid_cryptos():
    """Fetch the top 10 most liquid cryptocurrencies from Binance (excluding stablecoins)"""
    try:
        client = get_binance_client()
        tickers = client.get_ticker()
        
        usdt_pairs = [ticker for ticker in tickers if ticker['symbol'].endswith('USDT')]
        usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        
        stablecoin_keywords = ['BUSD', 'USDC', 'DAI', 'TUSD', 'PAX', 'USDP']
        liquid_cryptos = []
        
        for pair in usdt_pairs:
            symbol = pair['symbol'].replace('USDT', '')
            
            if any(stable in symbol for stable in stablecoin_keywords):
                continue
            
            if any(crypto['symbol'] == symbol for crypto in liquid_cryptos):
                continue
            
            liquid_cryptos.append({
                'symbol': symbol,
                'base_asset': symbol,
                'quote_asset': 'USDT',
                'full_symbol': pair['symbol'],
                'volume': float(pair['quoteVolume'])
            })
            
            if len(liquid_cryptos) >= 10:
                break
        
        # Add Bitcoin as the first asset if not already included
        btc_exists = any(crypto['symbol'] == 'BTC' for crypto in liquid_cryptos)
        if not btc_exists:
            btc_pair = next((p for p in usdt_pairs if p['symbol'] == 'BTCUSDT'), None)
            if btc_pair:
                liquid_cryptos.insert(0, {
                    'symbol': 'BTC',
                    'base_asset': 'BTC',
                    'quote_asset': 'USDT',
                    'full_symbol': 'BTCUSDT',
                    'volume': float(btc_pair['quoteVolume'])
                })
        
        return liquid_cryptos[:11]
    
    except Exception as e:
        print(f"Error fetching liquid cryptos from Binance: {e}")
        return [
            {'symbol': 'BTC', 'base_asset': 'BTC', 'quote_asset': 'USDT', 'full_symbol': 'BTCUSDT'},
            {'symbol': 'ETH', 'base_asset': 'ETH', 'quote_asset': 'USDT', 'full_symbol': 'ETHUSDT'},
            {'symbol': 'BNB', 'base_asset': 'BNB', 'quote_asset': 'USDT', 'full_symbol': 'BNBUSDT'},
            {'symbol': 'XRP', 'base_asset': 'XRP', 'quote_asset': 'USDT', 'full_symbol': 'XRPUSDT'},
            {'symbol': 'ADA', 'base_asset': 'ADA', 'quote_asset': 'USDT', 'full_symbol': 'ADAUSDT'},
            {'symbol': 'SOL', 'base_asset': 'SOL', 'quote_asset': 'USDT', 'full_symbol': 'SOLUSDT'},
            {'symbol': 'DOT', 'base_asset': 'DOT', 'quote_asset': 'USDT', 'full_symbol': 'DOTUSDT'},
            {'symbol': 'DOGE', 'base_asset': 'DOGE', 'quote_asset': 'USDT', 'full_symbol': 'DOGEUSDT'},
            {'symbol': 'AVAX', 'base_asset': 'AVAX', 'quote_asset': 'USDT', 'full_symbol': 'AVAXUSDT'},
            {'symbol': 'LINK', 'base_asset': 'LINK', 'quote_asset': 'USDT', 'full_symbol': 'LINKUSDT'},
            {'symbol': 'LTC', 'base_asset': 'LTC', 'quote_asset': 'USDT', 'full_symbol': 'LTCUSDT'}
        ]

def fetch_binance_historical_data(symbol, start_date, end_date):
    """Fetch historical daily price data from Binance"""
    try:
        client = get_binance_client()
        
        start_str = start_date.strftime("%d %b, %Y")
        end_str = end_date.strftime("%d %b, %Y")
        
        print(f"Fetching data for {symbol} from {start_str} to {end_str}")
        
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1DAY,
            start_str=start_str,
            end_str=end_str
        )
        
        if not klines:
            print(f"No data returned for {symbol}")
            return None
        
        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        for kline in klines:
            timestamp = datetime.datetime.fromtimestamp(kline[0] / 1000)
            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            volume = float(kline[5])
            
            dates.append(timestamp)
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
        
        df = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }).set_index('date')
        
        print(f"Fetched {len(df)} days of data for {symbol}")
        return df
    
    except Exception as e:
        print(f"Error fetching Binance data for {symbol}: {e}")
        return None

def create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators for the model features"""
    df = df.copy()
    
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['price_change'] = df['close'].diff()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Moving averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # Moving average ratios
    df['sma_5_ratio'] = df['close'] / df['sma_5']
    df['sma_10_ratio'] = df['close'] / df['sma_10']
    df['sma_20_ratio'] = df['close'] / df['sma_20']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=10).std()
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    # Target variable: 1 if next day return is positive, 0 otherwise
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)
    
    return df.dropna()

def train_logistic_regression_model(df: pd.DataFrame, symbol: str) -> Tuple[LogisticRegression, StandardScaler, dict]:
    """Train logistic regression model for price direction prediction"""
    try:
        # Select features (excluding target and date columns)
        feature_columns = [col for col in df.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        X = df[feature_columns]
        y = df['target']
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            C=0.1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        metrics = {
            'accuracy': accuracy,
            'feature_importance': feature_importance.head(10).to_dict('records'),
            'test_size': len(X_test),
            'train_size': len(X_train),
            'prediction_confidence': float(np.mean(np.max(y_pred_proba, axis=1)))
        }
        
        print(f"✅ Model trained for {symbol}: Accuracy = {accuracy:.4f}")
        
        return model, scaler, metrics
        
    except Exception as e:
        print(f"❌ Error training model for {symbol}: {e}")
        return None, None, {}

def predict_future_direction(model, scaler, df: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """Predict future price directions"""
    try:
        feature_columns = [col for col in df.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        X_recent = df[feature_columns].tail(days)
        
        if len(X_recent) < days:
            return pd.DataFrame()
        
        X_scaled = scaler.transform(X_recent)
        
        predictions = model.predict(X_scaled)
        prediction_proba = model.predict_proba(X_scaled)
        
        results = []
        for i in range(len(predictions)):
            confidence = prediction_proba[i][predictions[i]]
            direction = 'UP' if predictions[i] == 1 else 'DOWN'
            
            results.append({
                'day': i + 1,
                'prediction': predictions[i],
                'direction': direction,
                'confidence': confidence,
                'date': df.index[-days + i] if i < len(df.index) - days else df.index[-1] + pd.Timedelta(days=i)
            })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        return pd.DataFrame()

def calculate_relative_returns_and_covariance(cryptos_data):
    """Calculate relative price changes and covariance with Bitcoin"""
    combined_data = pd.DataFrame()
    
    for symbol, data in cryptos_data.items():
        combined_data[symbol] = data['close']
    
    returns_data = combined_data.pct_change().dropna()
    
    btc_returns = returns_data['BTC']
    covariance_data = {}
    
    for symbol in returns_data.columns:
        if symbol != 'BTC':
            cov = returns_data[symbol].cov(btc_returns)
            correlation = returns_data[symbol].corr(btc_returns)
            beta = cov / btc_returns.var()
            
            covariance_data[symbol] = {
                'covariance': cov,
                'correlation': correlation,
                'beta': beta,
                'volatility': returns_data[symbol].std()
            }
    
    covariance_data['BTC'] = {
        'covariance': 0,
        'correlation': 1,
        'beta': 1,
        'volatility': btc_returns.std()
    }
    
    return returns_data, covariance_data

def create_analysis_plots(returns_data, cryptos_info, predictions_data):
    """Create comprehensive analysis plots including predictions"""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplots
    gs = plt.GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])  # Cumulative returns
    ax2 = fig.add_subplot(gs[1, 0])  # Recent returns
    ax3 = fig.add_subplot(gs[1, 1])  # Predictions
    ax4 = fig.add_subplot(gs[2, :])  # Model performance
    
    # Plot 1: Cumulative returns
    cumulative_returns = (1 + returns_data).cumprod()
    colors = plt.cm.Set3(np.linspace(0, 1, len(cumulative_returns.columns)))
    
    for i, symbol in enumerate(cumulative_returns.columns):
        if symbol != 'BTC':
            ax1.plot(cumulative_returns.index, cumulative_returns[symbol], 
                    label=f'{symbol}', linewidth=2, color=colors[i])
    
    ax1.plot(cumulative_returns.index, cumulative_returns['BTC'], 
            label='BTC', linewidth=3, color='orange', linestyle='--')
    
    ax1.set_title('Cumulative Relative Price Changes (Jan 2022 - Sep 2023)', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Returns')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Recent daily returns
    recent_returns = returns_data.tail(30)
    symbols = [col for col in recent_returns.columns if col != 'BTC']
    
    for i, symbol in enumerate(symbols):
        ax2.plot(recent_returns.index, recent_returns[symbol] * 100, 
                label=f'{symbol}', alpha=0.7, linewidth=1.5, color=colors[i])
    
    ax2.plot(recent_returns.index, recent_returns['BTC'] * 100, 
            label='BTC', linewidth=2, color='orange')
    
    ax2.set_title('Recent Daily Returns (%) - Last 30 Days', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Daily Return (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Predictions
    if predictions_data:
        symbols = list(predictions_data.keys())
        prediction_days = 5
        
        # Create prediction bars
        bar_width = 0.8 / len(symbols)
        for i, symbol in enumerate(symbols):
            if symbol in predictions_data and not predictions_data[symbol].empty:
                pred_df = predictions_data[symbol]
                positions = np.arange(len(pred_df)) + i * bar_width
                
                colors_pred = ['#4CAF50' if pred == 1 else '#f44336' for pred in pred_df['prediction']]
                
                bars = ax3.bar(positions, pred_df['confidence'] * 100, 
                              width=bar_width, color=colors_pred, alpha=0.7,
                              label=symbol)
                
                # Add confidence labels
                for bar, conf in zip(bars, pred_df['confidence']):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{conf:.1%}', ha='center', va='bottom', fontsize=8)
        
        ax3.set_xlabel('Prediction Day')
        ax3.set_ylabel('Confidence (%)')
        ax3.set_title('Price Direction Predictions (Next 5 Days)', fontsize=12, fontweight='bold')
        ax3.set_xticks(np.arange(prediction_days) + bar_width * len(symbols) / 2)
        ax3.set_xticklabels([f'Day {i+1}' for i in range(prediction_days)])
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 110)
    
    # Plot 4: Model performance
    if models:
        symbols = list(models.keys())
        accuracies = [models[symbol]['metrics'].get('accuracy', 0) for symbol in symbols]
        confidences = [models[symbol]['metrics'].get('prediction_confidence', 0) for symbol in symbols]
        
        x = np.arange(len(symbols))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, accuracies, width, label='Accuracy', color='#2196F3', alpha=0.7)
        bars2 = ax4.bar(x + width/2, confidences, width, label='Avg Confidence', color='#FF9800', alpha=0.7)
        
        ax4.set_xlabel('Cryptocurrency')
        ax4.set_ylabel('Score')
        ax4.set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(symbols, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close()
    
    return img_bytes

def create_predictions_detail_plot(predictions_data):
    """Create detailed predictions plot"""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    if not predictions_data:
        return create_empty_plot("No prediction data available")
    
    symbols = list(predictions_data.keys())[:4]  # Show first 4 symbols
    
    for i, symbol in enumerate(symbols):
        if i >= len(axes):
            break
            
        if symbol in predictions_data and not predictions_data[symbol].empty:
            pred_df = predictions_data[symbol]
            
            # Create donut chart for predictions
            up_count = len(pred_df[pred_df['prediction'] == 1])
            down_count = len(pred_df[pred_df['prediction'] == 0])
            
            sizes = [up_count, down_count]
            colors = ['#4CAF50', '#f44336']
            labels = [f'UP ({up_count} days)', f'DOWN ({down_count} days)']
            
            wedges, texts, autotexts = axes[i].pie(sizes, colors=colors, startangle=90,
                                                  wedgeprops=dict(width=0.3), autopct='%1.1f%%')
            
            # Style the text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            axes[i].set_title(f'{symbol} - 5-Day Predictions\nAvg Confidence: {pred_df["confidence"].mean():.1%}', 
                            fontweight='bold', pad=20)
            
            # Add legend
            axes[i].legend(wedges, labels, title="Direction", loc="center left", 
                          bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close()
    
    return img_bytes

def create_empty_plot(message):
    """Create an empty plot with a message"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes, fontsize=16)
    ax.set_facecolor('#1e1e1e')
    ax.set_xticks([])
    ax.set_yticks([])
    
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close()
    
    return img_bytes

def update_data():
    """Background task to update crypto data and train models"""
    global latest_data, latest_covariance, latest_predictions, models, scalers
    
    while True:
        try:
            print("Updating crypto data from Binance and training models...")
            
            start_date = datetime.datetime(2022, 1, 1)
            end_date = datetime.datetime(2023, 9, 30)
            
            cryptos = get_most_liquid_cryptos()
            print(f"Fetching data for: {[crypto['symbol'] for crypto in cryptos]}")
            
            cryptos_data = {}
            successful_fetches = 0
            
            for crypto in cryptos:
                print(f"Fetching {crypto['symbol']}...")
                data = fetch_binance_historical_data(crypto['full_symbol'], start_date, end_date)
                if data is not None and not data.empty:
                    cryptos_data[crypto['symbol']] = data
                    successful_fetches += 1
                    print(f"✓ Successfully fetched {len(data)} records for {crypto['symbol']}")
                else:
                    print(f"✗ Failed to fetch data for {crypto['symbol']}")
                
                time.sleep(0.5)
            
            if successful_fetches > 1:
                # Calculate returns and covariance
                returns_data, covariance_data = calculate_relative_returns_and_covariance(cryptos_data)
                
                # Train models and make predictions
                predictions_data = {}
                models = {}
                scalers = {}
                
                for symbol, data in cryptos_data.items():
                    print(f"Training model for {symbol}...")
                    
                    # Create technical indicators
                    df_with_indicators = create_technical_indicators(data)
                    
                    if len(df_with_indicators) > 50:  # Minimum data requirement
                        model, scaler, metrics = train_logistic_regression_model(df_with_indicators, symbol)
                        
                        if model is not None:
                            models[symbol] = {
                                'model': model,
                                'scaler': scaler,
                                'metrics': metrics
                            }
                            scalers[symbol] = scaler
                            
                            # Make predictions
                            predictions = predict_future_direction(model, scaler, df_with_indicators)
                            predictions_data[symbol] = predictions
                            
                            print(f"✓ Predictions for {symbol}: {list(predictions['direction'])}")
                    
                    time.sleep(0.1)  # Small delay between model training
                
                latest_data = {
                    'returns': returns_data,
                    'cryptos_info': cryptos,
                    'covariance_data': covariance_data,
                    'last_update': datetime.datetime.now(),
                    'models_trained': len(models)
                }
                latest_covariance = covariance_data
                latest_predictions = predictions_data
                
                print(f"✅ Update completed! {successful_fetches} assets, {len(models)} models trained")
                
            else:
                print("Insufficient data fetched")
                
        except Exception as e:
            print(f"Error updating data: {e}")
            import traceback
            traceback.print_exc()
        
        print("Waiting 2 hours for next update...")
        time.sleep(7200)

# Flask Routes
@app.route('/')
def index():
    """Main page with comprehensive analysis"""
    global latest_data, latest_predictions
    
    if latest_data is None:
        return """
        <html>
            <head><title>Crypto Analysis with ML</title>
            <style>
                body { background-color: #1e1e1e; color: white; font-family: Arial; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; text-align: center; }
                .loading { color: #4CAF50; font-size: 18px; margin-top: 50px; }
            </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 Crypto Analysis with Machine Learning</h1>
                    <p>Analyzing 10 most liquid assets with Logistic Regression predictions</p>
                    <div class="loading">
                        <p>Data is being loaded and models are being trained...</p>
                        <p>This may take a few minutes for the initial load.</p>
                    </div>
                </div>
            </body>
        </html>
        """
    
    img_bytes = create_analysis_plots(latest_data['returns'], latest_data['cryptos_info'], latest_predictions)
    return send_file(img_bytes, mimetype='image/png')

@app.route('/predictions')
def predictions_detail():
    """Detailed predictions page"""
    global latest_predictions
    
    if not latest_predictions:
        return send_file(create_empty_plot("No prediction data available"), mimetype='image/png')
    
    img_bytes = create_predictions_detail_plot(latest_predictions)
    return send_file(img_bytes, mimetype='image/png')

@app.route('/data')
def get_data():
    """API endpoint to get all data"""
    global latest_data, latest_predictions, models
    
    if latest_data is None:
        return jsonify({"error": "Data not available yet"}), 503
    
    response_data = {
        "covariance_data": latest_covariance,
        "last_updated": datetime.datetime.now().isoformat(),
        "data_source": "Binance",
        "models_trained": latest_data.get('models_trained', 0),
        "predictions": {}
    }
    
    if latest_predictions:
        for symbol, pred_df in latest_predictions.items():
            if not pred_df.empty:
                response_data["predictions"][symbol] = pred_df.to_dict('records')
    
    if models:
        response_data["model_metrics"] = {
            symbol: data['metrics'] for symbol, data in models.items()
        }
    
    return jsonify(response_data)

@app.route('/status')
def status():
    """Status page"""
    global latest_data, latest_predictions, models
    
    status_info = {
        "status": "running",
        "data_source": "Binance API",
        "last_update": datetime.datetime.now().isoformat(),
        "assets_loaded": len(latest_data['cryptos_info']) if latest_data else 0,
        "models_trained": latest_data.get('models_trained', 0) if latest_data else 0
    }
    
    html = f"""
    <html>
        <head>
            <title>Status - Crypto ML Analysis</title>
            <style>
                body {{ background-color: #1e1e1e; color: white; font-family: Arial; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .card {{ background-color: #2d2d2d; padding: 20px; margin: 10px 0; border-radius: 8px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #444; }}
                th {{ background-color: #3d3d3d; }}
                a {{ color: #4CAF50; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                .positive {{ color: #4CAF50; }}
                .negative {{ color: #f44336; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔮 Crypto ML Analysis - Status</h1>
                
                <div class="card">
                    <h2>System Status</h2>
                    <p><strong>Status:</strong> <span style="color: #4CAF50;">{status_info['status'].upper()}</span></p>
                    <p><strong>Data Source:</strong> {status_info['data_source']}</p>
                    <p><strong>Last Update:</strong> {status_info['last_update']}</p>
                    <p><strong>Assets Loaded:</strong> {status_info['assets_loaded']}</p>
                    <p><strong>Models Trained:</strong> {status_info['models_trained']}</p>
                </div>
    """
    
    if latest_predictions and models:
        html += """
                <div class="card">
                    <h2>Latest Predictions (Next 5 Days)</h2>
                    <table>
                        <tr>
                            <th>Asset</th>
                            <th>Accuracy</th>
                            <th>Day 1</th>
                            <th>Day 2</th>
                            <th>Day 3</th>
                            <th>Day 4</th>
                            <th>Day 5</th>
                        </tr>
        """
        
        for symbol in latest_predictions.keys():
            if symbol in models and symbol in latest_predictions and not latest_predictions[symbol].empty:
                accuracy = models[symbol]['metrics'].get('accuracy', 0)
                pred_df = latest_predictions[symbol]
                
                html += f"""
                        <tr>
                            <td><strong>{symbol}</strong></td>
                            <td>{accuracy:.3f}</td>
                """
                
                for _, pred in pred_df.iterrows():
                    direction_class = "positive" if pred['direction'] == 'UP' else "negative"
                    html += f'<td class="{direction_class}">{pred["direction"]} ({pred["confidence"]:.1%})</td>'
                
                html += "</tr>"
        
        html += """
                    </table>
                </div>
        """
    
    html += """
                <div class="card">
                    <h2>Navigation</h2>
                    <p>
                        <a href="/">📈 View Main Analysis</a> | 
                        <a href="/predictions">🔮 View Detailed Predictions</a> | 
                        <a href="/data">🔗 Raw Data (JSON)</a>
                    </p>
                </div>
            </div>
        </body>
    </html>
    """
    
    return html

if __name__ == '__main__':
    update_thread = Thread(target=update_data, daemon=True)
    update_thread.start()
    
    print("Starting Crypto ML Analysis Server...")
    print("Initial data loading and model training may take a few minutes.")
    print("Server will be available at http://0.0.0.0:8080")
    time.sleep(2)
    
    app.run(host='0.0.0.0', port=8080, debug=False)

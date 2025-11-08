"""
Tick Data Loader and Market Data Downloader
===========================================

Handles real market data download from Twelve Data API
and tick-level data ingestion for analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import warnings
import configparser
import os
from twelvedata import TDClient
import time


def load_ticks(path: str, file_format: str = 'csv') -> pd.DataFrame:
    """
    Load tick-level FX data from file.
    
    Args:
        path: Path to data file
        file_format: 'csv', 'parquet', or 'hdf5'
        
    Returns:
        DataFrame with columns: timestamp, bid, ask, last, volume
        Sorted by timestamp, duplicates removed
    """
    if file_format == 'csv':
        df = pd.read_csv(path)
    elif file_format == 'parquet':
        df = pd.read_parquet(path)
    elif file_format == 'hdf5':
        df = pd.read_hdf(path)
    else:
        raise ValueError(f"Unsupported file_format: {file_format}")
    
    # Ensure required columns
    required = ['timestamp', 'bid', 'ask']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing required columns. Need: {required}")
    
    # Parse timestamp
    if df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif df['timestamp'].dtype in ['int64', 'float64']:
        # Assume epoch milliseconds
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Add 'last' if missing (mid-price)
    if 'last' not in df.columns:
        df['last'] = (df['bid'] + df['ask']) / 2
    
    # Add volume if missing
    if 'volume' not in df.columns:
        df['volume'] = 1.0
    
    # Clean data
    df = df.sort_values('timestamp')
    df = df.drop_duplicates(subset='timestamp', keep='last')
    df = df.reset_index(drop=True)
    
    # Remove NaN
    df = df.dropna(subset=['bid', 'ask', 'last'])
    
    print(f"Loaded {len(df)} ticks from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def download_real_data(
    pair: str = 'AUD/USD',
    days: int = 90,
    interval: str = '1min',
    output_file: str = None,
    config_file: str = '../config.ini'
) -> str:
    """
    Download real FX market data from Twelve Data API
    
    Args:
        pair: Currency pair (e.g., 'AUD/USD', 'USD/CAD')
        days: Number of days of historical data (max depends on API tier)
        interval: Data interval ('1min', '5min', '15min', '1h')
        output_file: Path to save CSV (auto-generated if None)
        config_file: Path to config.ini with API key
        
    Returns:
        Path to saved CSV file
    """
    print(f"\n📊 Downloading {days} days of {pair} data ({interval} intervals)...")
    
    # Load API key
    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file)
    else:
        # Try parent directory
        parent_config = os.path.join(os.path.dirname(__file__), '..', 'config.ini')
        if os.path.exists(parent_config):
            config.read(parent_config)
        else:
            raise FileNotFoundError(
                "config.ini not found. Please create one with:\n"
                "[TWELVE_DATA]\n"
                "api_key = YOUR_API_KEY_HERE"
            )
    
    if not config.has_option('TWELVE_DATA', 'api_key'):
        raise ValueError("config.ini must have [TWELVE_DATA] api_key")
    
    api_key = config['TWELVE_DATA']['api_key']
    client = TDClient(apikey=api_key)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Fetching data...")
    
    try:
        # Get time series data
        ts = client.time_series(
            symbol=pair,
            interval=interval,
            outputsize=5000,  # Max per request
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            timezone='UTC'
        )
        
        df = ts.as_pandas()
        
        if df.empty:
            raise ValueError(f"No data returned for {pair}")
        
        # Convert to tick format (timestamp, bid, ask, last, volume)
        df = df.reset_index()
        df = df.rename(columns={'datetime': 'timestamp'})
        
        # Convert OHLC to bid/ask/last
        # Use close as 'last', create bid/ask with small spread
        spread = 0.0001  # 1 pip spread
        df['last'] = df['close']
        df['bid'] = df['close'] - spread / 2
        df['ask'] = df['close'] + spread / 2
        
        # Keep relevant columns
        df = df[['timestamp', 'bid', 'ask', 'last', 'volume']]
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        df = df.reset_index(drop=True)
        
        print(f"  ✓ Downloaded {len(df)} data points")
        print(f"  Price range: {df['last'].min():.6f} - {df['last'].max():.6f}")
        
        # Save to CSV
        if output_file is None:
            pair_clean = pair.replace('/', '')
            output_file = f"data/{pair_clean}_{interval}_{days}days.csv"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        df.to_csv(output_file, index=False)
        print(f"  ✓ Saved to: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        raise


def generate_synthetic_ticks(
    n_ticks: int = 100000,
    base_price: float = 1.0,
    spread: float = 0.0001,
    regime: str = 'brownian',
    theta: float = 0.1,  # Mean reversion speed
    sigma: float = 0.0005,  # Volatility
    jump_prob: float = 0.001,  # Jump probability
    jump_size: float = 0.002,  # Jump magnitude
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic tick data with configurable market regimes.
    
    Regimes:
        - 'brownian': Geometric Brownian motion (trending)
        - 'mean_reversion': Ornstein-Uhlenbeck process
        - 'jump': Brownian + Poisson jumps
        - 'volatility_clustering': GARCH-like dynamics
        - 'mixed': Regime-switching between all
        
    Args:
        n_ticks: Number of ticks to generate
        base_price: Starting price
        spread: Bid-ask spread
        regime: Market regime type
        theta: Mean reversion parameter
        sigma: Volatility parameter
        jump_prob: Probability of jumps
        jump_size: Jump magnitude
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with timestamp, bid, ask, last, volume columns
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate timestamps (high-frequency, realistic tick spacing)
    start_time = datetime(2025, 1, 1)
    # Generate realistic tick intervals (average 1-10 seconds between ticks)
    intervals = np.random.exponential(scale=3.0, size=n_ticks)  # Average 3 seconds
    intervals = np.maximum(intervals, 0.1)  # Minimum 0.1 seconds
    timestamps = [start_time + timedelta(seconds=sum(intervals[:i+1])) for i in range(n_ticks)]
    
    # Initialize price array
    prices = np.zeros(n_ticks)
    prices[0] = base_price
    
    # Generate price dynamics based on regime
    if regime == 'brownian':
        # Geometric Brownian Motion
        dt = 3.0 / (252 * 24 * 60 * 60)  # 3 seconds in years (high-frequency)
        for i in range(1, n_ticks):
            dW = np.random.normal(0, np.sqrt(dt))
            prices[i] = prices[i-1] * np.exp((0.0 - 0.5*sigma**2)*dt + sigma*dW)
    
    elif regime == 'mean_reversion':
        # Ornstein-Uhlenbeck process
        dt = 3.0 / (252 * 24 * 60 * 60)  # 3 seconds in years (high-frequency)
        for i in range(1, n_ticks):
            dW = np.random.normal(0, np.sqrt(dt))
            prices[i] = prices[i-1] + theta*(base_price - prices[i-1])*dt + sigma*dW
    
    elif regime == 'jump':
        # Brownian + Poisson jumps
        dt = 3.0 / (252 * 24 * 60 * 60)  # 3 seconds in years (high-frequency)
        for i in range(1, n_ticks):
            dW = np.random.normal(0, np.sqrt(dt))
            jump = jump_size * np.random.randn() if np.random.random() < jump_prob else 0
            prices[i] = prices[i-1] * np.exp((0.0 - 0.5*sigma**2)*dt + sigma*dW + jump)
    
    elif regime == 'volatility_clustering':
        # GARCH(1,1)-like dynamics
        dt = 3.0 / (252 * 24 * 60 * 60)  # 3 seconds in years (high-frequency)
        vol = np.zeros(n_ticks)
        vol[0] = sigma
        
        alpha = 0.1  # ARCH term
        beta = 0.85  # GARCH term
        omega = sigma**2 * (1 - alpha - beta)
        
        for i in range(1, n_ticks):
            # Update volatility
            returns_sq = ((prices[i-1] - prices[max(0, i-2)]) / prices[max(0, i-2)])**2 if i > 1 else 0
            vol[i] = np.sqrt(omega + alpha*returns_sq + beta*vol[i-1]**2)
            
            # Generate return
            dW = np.random.normal(0, np.sqrt(dt))
            prices[i] = prices[i-1] * np.exp(-0.5*vol[i]**2*dt + vol[i]*dW)
    
    elif regime == 'mixed':
        # Regime switching (Markov chain)
        regimes = np.random.choice([0, 1, 2], size=n_ticks, p=[0.4, 0.4, 0.2])
        dt = 3.0 / (252 * 24 * 60 * 60)  # 3 seconds in years (high-frequency)
        
        for i in range(1, n_ticks):
            dW = np.random.normal(0, np.sqrt(dt))
            
            if regimes[i] == 0:  # Brownian
                prices[i] = prices[i-1] * np.exp((0.0 - 0.5*sigma**2)*dt + sigma*dW)
            elif regimes[i] == 1:  # Mean reversion
                prices[i] = prices[i-1] + theta*(base_price - prices[i-1])*dt + sigma*dW
            else:  # Jump
                jump = jump_size * np.random.randn() if np.random.random() < jump_prob*3 else 0
                prices[i] = prices[i-1] * np.exp((0.0 - 0.5*sigma**2)*dt + sigma*dW + jump)
    
    else:
        raise ValueError(f"Unknown regime: {regime}")
    
    # Ensure positive prices
    prices = np.abs(prices)
    prices = np.where(prices < base_price * 0.5, base_price * 0.5, prices)
    prices = np.where(prices > base_price * 2.0, base_price * 2.0, prices)
    
    # Generate bid/ask from mid-price
    half_spread = spread / 2
    bid = prices - half_spread
    ask = prices + half_spread
    
    # Generate volume (Poisson-distributed)
    volume = np.random.poisson(lam=100, size=n_ticks).astype(float)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'bid': bid,
        'ask': ask,
        'last': prices,
        'volume': volume
    })
    
    print(f"Generated {n_ticks} synthetic ticks ({regime} regime)")
    print(f"  Price range: {prices.min():.6f} - {prices.max():.6f}")
    print(f"  Mean: {prices.mean():.6f}, Std: {prices.std():.6f}")
    
    return df


def resample_ticks(ticks: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    """
    Resample ticks to regular intervals (optional utility).
    
    Args:
        ticks: Tick DataFrame
        freq: Pandas frequency string ('1min', '5min', etc.)
        
    Returns:
        Resampled DataFrame
    """
    ticks = ticks.set_index('timestamp')
    
    resampled = ticks.resample(freq).agg({
        'bid': 'last',
        'ask': 'last',
        'last': 'last',
        'volume': 'sum'
    })
    
    resampled = resampled.dropna()
    resampled = resampled.reset_index()
    
    return resampled


if __name__ == "__main__":
    # Test synthetic data generation
    print("=" * 80)
    print("Testing Synthetic Tick Generation")
    print("=" * 80)
    
    for regime in ['brownian', 'mean_reversion', 'jump', 'volatility_clustering', 'mixed']:
        df = generate_synthetic_ticks(
            n_ticks=10000,
            regime=regime,
            seed=42
        )
        print()

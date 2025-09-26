import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from hurst import compute_Hc
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def download_and_analyze_nifty50_data():
    """
    This script downloads historical data for the NIFTY 50 index,
    calculates its rolling Hurst exponent, and performs a basic analysis
    of its market behavior segments (trending vs. mean-reverting).

    It is adapted from the methodology used in the S&P 500 analysis notebooks
    and corrected for the NIFTY 50 ticker's data structure.
    """
    # --- 1. Data Download and Setup ---
    print("--- Starting Data Collection and Analysis for NIFTY 50 ---")
    
    # User-defined parameters
    index_ticker = "^NSEI"
    index_name = "NIFTY 50"
    start_date = '2010-01-01'
    end_date = '2024-12-31'
    hurst_window = 100
    
    # Create necessary directories to save data and results
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # --- 2. Download and Prepare Data ---
    print(f"\nDownloading {index_name} data ({index_ticker})...")
    try:
        # Download the historical data using yfinance
        df = yf.download(index_ticker, start=start_date, end=end_date)
        if df.empty:
            raise ValueError("No data returned for the specified ticker.")
        
        # **CORRECTED SECTION**
        # Flatten MultiIndex columns if present (e.g., ('Close', '^NSEI'))
        # This resolves the MergeError and potential KeyErrors.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Save the raw data to a CSV file
        df.to_csv(f'data/nifty50_data.csv')
        print(f"Data downloaded successfully. Total rows: {df.shape[0]}")

    except Exception as e:
        print(f"An error occurred during data download: {e}")
        return

    # Ensure the 'Close' column is used, as 'Adj Close' may not always be present
    if 'Adj Close' in df.columns and not df['Adj Close'].isnull().all():
        close_col = 'Adj Close'
    else:
        close_col = 'Close'
    print(f"Using '{close_col}' column for analysis.")
        
    # Calculate log returns for Hurst exponent calculation
    df['log_returns'] = np.log(df[close_col] / df[close_col].shift(1))
    df.dropna(inplace=True)
    
    # Save log returns for potential later use
    df['log_returns'].to_csv(f'data/nifty50_log_returns.csv')

    # --- 3. Hurst Exponent Calculation ---
    print(f"\nCalculating {hurst_window}-day rolling Hurst exponent...")
    
    log_returns = df['log_returns']
    hurst_values = []
    dates = []

    for i in range(hurst_window, len(log_returns)):
        data_window = log_returns.iloc[i - hurst_window:i].values
        try:
            # Calculate Hurst exponent for the window
            H, _, _ = compute_Hc(data_window, kind='change', simplified=True)
            hurst_values.append(H)
            dates.append(log_returns.index[i])
        except Exception:
            # Some windows might not have enough data to compute H
            continue
            
    hurst_series = pd.Series(hurst_values, index=dates, name=f'hurst_{hurst_window}')
    hurst_series.to_csv(f'data/nifty50_hurst_{hurst_window}.csv')
    print(f"Hurst exponent calculation complete. {len(hurst_values)} values calculated.")

    # --- 4. Data Analysis and Visualization ---
    print("\nGenerating analysis plots...")

    # Plot 1: NIFTY 50 Price
    plt.figure(figsize=(12, 6))
    plt.plot(df[close_col], label=close_col)
    plt.title(f'{index_name} Historical Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/nifty50_price_chart.png')
    plt.show()

    # Plot 2: Hurst Exponent Over Time
    plt.figure(figsize=(12, 7))
    plt.plot(hurst_series, label=f'Hurst {hurst_window}-day', color='blue')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Random Walk (H=0.5)')
    plt.title(f'{index_name} - Hurst Exponent Over Time')
    plt.ylabel('Hurst Exponent')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/nifty50_hurst_analysis.png')
    plt.show()

    # --- 5. Segment Analysis ---
    # Prepare data for comparing Hurst to market trend
    analysis_df = df.join(hurst_series, how='inner')
    analysis_df['MA20'] = analysis_df[close_col].rolling(window=20).mean()
    analysis_df['MA20_diff'] = analysis_df['MA20'].diff()
    
    analysis_df['Increasing'] = (analysis_df['MA20_diff'] > 0).astype(int)
    analysis_df['Decreasing'] = (analysis_df['MA20_diff'] < 0).astype(int)
    
    # Calculate trend metrics over the same window as Hurst
    analysis_df['Increasing_days'] = analysis_df['Increasing'].rolling(window=hurst_window).sum()
    analysis_df['Decreasing_days'] = analysis_df['Decreasing'].rolling(window=hurst_window).sum()
    
    analysis_df['Trend_Ratio'] = analysis_df['Increasing_days'] / analysis_df['Decreasing_days']
    analysis_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    analysis_df.dropna(inplace=True)

    # Plot 3: Hurst Exponent vs. Trend Ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(analysis_df[f'hurst_{hurst_window}'], analysis_df['Trend_Ratio'], alpha=0.5)
    plt.xlabel(f"Hurst Exponent (window={hurst_window})")
    plt.ylabel("Ratio of Up-Trend to Down-Trend Days")
    plt.title(f"Hurst Exponent vs. Market Trend for {index_name}")
    plt.axhline(y=1.0, color='red', linestyle='--', label='Balanced Trend (Ratio=1.0)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/nifty50_hurst_vs_trend_ratio.png')
    plt.show()

    print("\n--- Process Complete ---")
    print("All data and plots have been saved in the 'data/' and 'results/' directories.")


if __name__ == '__main__':
    download_and_analyze_nifty50_data()
import pandas as pd
import numpy as np
from hurst import compute_Hc
import matplotlib.pyplot as plt
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def calculate_rolling_hurst(returns, window=100):
    """
    Calculates the rolling Hurst exponent for a given time series of returns.
    """
    print(f"Calculating rolling Hurst exponent with a {window}-day window...")
    hurst_values = []
    dates = []
    
    # Ensure there's enough data to start the rolling window
    if len(returns) < window:
        print("Error: Not enough data to calculate Hurst exponent with the specified window.")
        return None

    for i in range(window, len(returns)):
        # Create the data window for the calculation
        data_window = returns.iloc[i - window:i].values.flatten()
        try:
            # Compute the Hurst exponent using the 'hurst' library
            H, _, _ = compute_Hc(data_window, kind='change', simplified=True)
            hurst_values.append(H)
            dates.append(returns.index[i])
        except Exception as e:
            # This handles cases where a window might be invalid (e.g., all zeros)
            # print(f"Warning: Could not calculate Hurst for window ending at {returns.index[i]}: {e}")
            continue
            
    if not hurst_values:
        print("Error: Hurst exponent calculation resulted in no valid values.")
        return None
        
    hurst_series = pd.Series(hurst_values, index=dates, name=f'hurst_{window}')
    print(f"Calculation complete. {len(hurst_values)} Hurst values were generated.")
    return hurst_series


def create_hurst_analysis_for_nifty50():
    """
    Loads the NIFTY 50 data, calculates the Hurst exponent, and generates
    an analysis plot, similar to the S&P 500 project's second step.
    """
    print("\n--- Starting Step 2: Hurst Exponent Calculation for NIFTY 50 ---")

    # --- 1. Load Data ---
    try:
        nifty_data_path = 'data/nifty50_data.csv'
        log_returns_path = 'data/nifty50_log_returns.csv'
        
        nifty_df = pd.read_csv(nifty_data_path, index_col=0, parse_dates=True)
        log_returns_series = pd.read_csv(log_returns_path, index_col=0, parse_dates=True).iloc[:, 0]
        
        print("Successfully loaded nifty50_data.csv and nifty50_log_returns.csv.")

    except FileNotFoundError as e:
        print(f"Error: Could not find required data files. Please run data_collection.py first.")
        print(e)
        return
        
    # --- 2. Calculate and Save Hurst Exponent ---
    hurst_100 = calculate_rolling_hurst(log_returns_series, 100)
    
    if hurst_100 is not None:
        hurst_100.to_csv('data/nifty50_hurst_100.csv')
        print("Hurst exponent data saved to 'data/nifty50_hurst_100.csv'.")
        
        # --- 3. Create Visualization ---
        print("Generating Hurst analysis plot...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot NIFTY 50 Price
        ax1.plot(nifty_df['Close'], label='NIFTY 50 Close Price')
        ax1.set_title('NIFTY 50 Price and Rolling Hurst Exponent')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Hurst Exponent
        ax2.plot(hurst_100, label='Hurst (100-day window)', color='orange')
        ax2.axhline(y=0.5, color='red', linestyle='--', label='Random Walk (H=0.5)')
        ax2.axhline(y=0.6, color='green', linestyle=':', label='Trending Threshold (H > 0.5)')
        ax2.axhline(y=0.4, color='purple', linestyle=':', label='Mean-Reverting Threshold (H < 0.5)')
        ax2.set_title('Hurst Exponent Over Time')
        ax2.set_ylabel('Hurst Value')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/nifty50_hurst_exponent_analysis.png')
        plt.show()

        # --- 4. Print Summary Statistics ---
        print("\n--- Hurst Exponent Statistics ---")
        print(f"Mean: {hurst_100.mean():.4f}")
        print(f"Std Dev:  {hurst_100.std():.4f}")
        print(f"Min:  {hurst_100.min():.4f}")
        print(f"Max:  {hurst_100.max():.4f}")
    
    print("\n--- Step 2 Complete ---")


if __name__ == "__main__":
    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)
    create_hurst_analysis_for_nifty50()
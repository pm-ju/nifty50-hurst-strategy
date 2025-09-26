import pandas as pd
import numpy as np
import ta
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_features_and_target_for_nifty50():
    """
    This script performs feature engineering for the NIFTY 50 project.
    It loads the data, creates market segments based on the Hurst exponent,
    generates the target variable for trading signals, and adds numerous
    technical indicators as features.
    """
    print("\n--- Starting Step 3 (Corrected): Feature Engineering for NIFTY 50 ---")

    # --- 1. Load Pre-processed Data ---
    try:
        data_path = 'data/nifty50_data.csv'
        hurst_path = 'data/nifty50_hurst_100.csv'
        
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        hurst_series = pd.read_csv(hurst_path, index_col=0, parse_dates=True)
        
        print("Successfully loaded data and Hurst exponent files.")
    except FileNotFoundError as e:
        print(f"Error: Data file not found. Please run previous steps first.")
        print(e)
        return

    # --- 2. Create Market Segments based on Hurst Exponent ---
    print("Creating market segments based on Hurst exponent...")
    
    # Join Hurst data with the main dataframe
    df = df.join(hurst_series, how='inner')
    
    hurst_cutoff = 0.60 
    
    df['Segment'] = np.where(df['hurst_100'] > hurst_cutoff, "Trending", "Mean Reverting")
    print(f"Segmentation complete. Cutoff used: H > {hurst_cutoff} for 'Trending'.")
    
    # --- 3. Create the Target (Dependent) Variable ---
    print("\nCreating the short-term target variable (Buy/Sell/Hold)...")
    
    std_window = 20
    analysis_window = 5

    # Calculate rolling standard deviation
    df['DVT_STD'] = df['Close'].rolling(std_window).std()
    
    df['DVT_MAX'] = df['Close'].rolling(analysis_window).max().shift(-analysis_window)
    df['DVT_MIN'] = df['Close'].rolling(analysis_window).min().shift(-analysis_window)

    df['DVT_Upper'] = df['Close'] + 1.0 * df['DVT_STD']
    df['DVT_Lower'] = df['Close'] - 1.5 * df['DVT_STD']

    df['Target'] = np.where(
        df['DVT_MAX'] > df['DVT_Upper'], 1,
        np.where(df['DVT_MIN'] < df['DVT_Lower'], -1, 0)
    )
    print("Target variable created.")
    
    # --- 4. Add Technical Analysis Features ---
    print("\nAdding technical analysis features using 'ta' library...")
    try:
        df = ta.add_all_ta_features(
            df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            fillna=True
        )
        print("Technical features added successfully.")
    except Exception as e:
        print(f"An error occurred during feature generation: {e}")

    # --- 5. Final Data Cleaning and Saving ---
    print("\nCleaning and saving the final dataset...")

    # **CORRECTION**: Only drop intermediate columns used for target *calculation*.
    # Keep 'DVT_STD', 'Open', and 'Close' as they are needed for later steps.
    df.drop(['DVT_MAX', 'DVT_MIN', 'DVT_Upper', 'DVT_Lower'], axis=1, inplace=True)
    
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    final_rows = df.shape[0]
    print(f"Dropped {initial_rows - final_rows} rows with NaN values.")

    # Save the final dataset
    df.to_csv('data/nifty50_final_dataset.csv')
    
    print("\n--- Step 3 (Corrected) Complete ---")
    print("Final dataset with features, segments, and target saved to 'data/nifty50_final_dataset.csv'.")
    print(f"Final dataset shape: {df.shape}")


if __name__ == "__main__":
    create_features_and_target_for_nifty50()
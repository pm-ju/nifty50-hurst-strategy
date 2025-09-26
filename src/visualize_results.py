import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_backtest(df):
    """
    Runs a simple, more robust backtest on the model's predictions.
    - Starts with $100 cash.
    - When a 'Buy' (1) signal appears, it invests all cash.
    - When a 'Sell' (-1) signal appears, it sells all shares and goes to cash.
    - It holds the position until the opposite signal appears or a hold (0) is signaled.
    """
    print("Running final backtest simulation...")
    
    initial_capital = 100.0
    # Use a copy to avoid SettingWithCopyWarning
    portfolio = df[['Close']].copy()
    portfolio['Signal'] = df['Predicted']
    portfolio['Position'] = 0.0
    portfolio['Portfolio_Value'] = initial_capital

    cash = initial_capital
    shares = 0.0
    
    for i in range(1, len(portfolio)):
        # Carry forward the previous day's state
        portfolio.loc[portfolio.index[i], 'Position'] = portfolio.loc[portfolio.index[i-1], 'Position']
        
        # BUY signal and not already in a position
        if portfolio.loc[portfolio.index[i], 'Signal'] == 1 and portfolio.loc[portfolio.index[i-1], 'Position'] == 0:
            shares = cash / df['Close'].iloc[i]
            cash = 0
            portfolio.loc[portfolio.index[i], 'Position'] = 1.0

        # SELL or HOLD signal and currently in a position
        elif portfolio.loc[portfolio.index[i], 'Signal'] != 1 and portfolio.loc[portfolio.index[i-1], 'Position'] == 1:
            cash = shares * df['Close'].iloc[i]
            shares = 0
            portfolio.loc[portfolio.index[i], 'Position'] = 0.0

        # Calculate portfolio value for the day
        if portfolio.loc[portfolio.index[i], 'Position'] == 1:
            portfolio.loc[portfolio.index[i], 'Portfolio_Value'] = shares * df['Close'].iloc[i]
        else:
            portfolio.loc[portfolio.index[i], 'Portfolio_Value'] = cash

    return portfolio

def calculate_and_print_metrics(df, period_name="Overall"):
    """Calculates and prints key performance metrics for the strategy."""
    
    returns = df['Portfolio_Value'].pct_change().dropna()
    
    if returns.empty:
        print(f"\n--- No trades were executed in the {period_name}. ---")
        return

    # Sharpe Ratio
    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    # Annual Return
    total_days = (df.index[-1] - df.index[0]).days
    annual_return = ((df['Portfolio_Value'].iloc[-1] / 100)**(365.0/total_days)) - 1 if total_days > 0 else 0
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()

    print(f"\n--- Performance Metrics: {period_name} ---")
    print(f"Annual Return:    {annual_return:.2%}")
    print(f"Sharpe Ratio:     {sharpe_ratio:.2f}")
    print(f"Max Drawdown:     {max_drawdown:.2%}")

def visualize_performance_pipeline():
    """Main pipeline to visualize the performance of the RandomForest strategy."""
    print("\n--- Starting Final Step: Performance Visualization (RandomForest) ---")
    
    try:
        # Load the combined predictions from the RandomForest model run
        predictions_path = 'data/nifty50_combined_rf_predictions.csv'
        df = pd.read_csv(predictions_path, index_col=0, parse_dates=True)
        print("Successfully loaded combined RandomForest predictions.")
    except FileNotFoundError:
        print(f"Error: '{predictions_path}' not found. Please run model_training.py first.")
        return

    # Split into Train and Test for Analysis
    df_train = df[df['Sample'] == 'Train'].copy()
    df_test = df[df['Sample'] == 'Test'].copy()

    # Run Trading Simulation
    portfolio_train = run_backtest(df_train)
    portfolio_test = run_backtest(df_test)

    # Calculate Buy and Hold Strategy for Comparison
    df_train['Buy_and_Hold'] = 100 * (df_train['Close'] / df_train['Close'].iloc[0])
    df_test['Buy_and_Hold'] = 100 * (df_test['Close'] / df_test['Close'].iloc[0])
    
    # Calculate and Print Performance Metrics
    calculate_and_print_metrics(portfolio_train, period_name="Training Period")
    calculate_and_print_metrics(portfolio_test, period_name="Testing Period")
    
    print("\nGenerating final performance plots...")

    # Plot 1: Training Period Performance
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_train['Portfolio_Value'], label='Hurst Segment RF Strategy')
    plt.plot(df_train['Buy_and_Hold'], label='Buy and Hold', linestyle='--')
    plt.title('Final NIFTY 50 Strategy Performance (RandomForest) - Training Period')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/performance_train_rf.png')
    plt.show()

    # Plot 2: Testing Period Performance
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_test['Portfolio_Value'], label='Hurst Segment RF Strategy')
    plt.plot(df_test['Buy_and_Hold'], label='Buy and Hold', linestyle='--')
    plt.title('Final NIFTY 50 Strategy Performance (RandomForest) - Testing Period')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/performance_test_rf.png')
    plt.show()

    # Save Final Simulation Data
    final_results_df = pd.concat([
        df_train.join(portfolio_train, rsuffix='_Portfolio'),
        df_test.join(portfolio_test, rsuffix='_Portfolio')
    ])
    final_results_df.to_csv('results/nifty50_final_simulation_results.csv')
    
    print("\n--- Project Complete ---")
    print("Final simulation results and plots saved in the 'results/' directory.")

if __name__ == "__main__":
    visualize_performance_pipeline()
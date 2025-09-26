import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report
import os
import pickle
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def split_data_xy(df, target_column='Target'):
    """Splits the dataframe into features (X) and target (y)."""
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    # Remap target labels from [-1, 0, 1] to [0, 1, 2]
    y_remapped = y + 1
    return X, y_remapped

def train_random_forest_model(X, y, segment_name):
    """
    Trains and evaluates a RandomForestClassifier using TimeSeriesSplit cross-validation.
    """
    print(f"\n--- Training RandomForest Model for '{segment_name}' Segment ---")

    # --- 1. Temporal Split (for final evaluation) ---
    # We'll use this split *after* finding the best model with cross-validation
    train_end_year = 2018
    X_train = X[X.index.year <= train_end_year]
    y_train = y[y.index.year <= train_end_year]
    X_test = X[X.index.year > train_end_year]
    y_test = y[y.index.year > train_end_year]

    # Drop non-feature columns for model training
    non_feature_cols = ['Open', 'Close', 'DVT_STD']
    X_train_model = X_train.drop(columns=non_feature_cols)
    X_test_model = X_test.drop(columns=non_feature_cols)

    # --- 2. Hyperparameter Grid for RandomForest ---
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10]
    }

    # --- 3. Model and TimeSeriesSplit Grid Search ---
    # TimeSeriesSplit ensures we always train on past data and test on future data
    tscv = TimeSeriesSplit(n_splits=5)
    clf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )

    # --- 4. Train the Model ---
    print("Running GridSearchCV with TimeSeriesSplit...")
    # We fit on the training portion of the data
    grid_search.fit(X_train_model, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")

    # --- 5. Evaluate Performance ---
    print("\n--- Model Performance ---")
    
    y_train_pred = best_model.predict(X_train_model)
    print("\nTraining Set Classification Report:")
    print(classification_report(y_train - 1, y_train_pred - 1, zero_division=0))
    
    y_test_pred = best_model.predict(X_test_model)
    print("\nTesting Set Classification Report:")
    print(classification_report(y_test - 1, y_test_pred - 1, zero_division=0))

    # --- 6. Save Model and Predictions ---
    model_filename = f'models/{segment_name}_nifty50_rf_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nTrained model saved to '{model_filename}'")
    
    # Create predictions for the original [-1, 0, 1] labels
    df_train_out = pd.DataFrame(y_train - 1, columns=['Target'])
    df_train_out['Predicted'] = y_train_pred - 1
    df_train_out['Sample'] = 'Train'

    df_test_out = pd.DataFrame(y_test - 1, columns=['Target'])
    df_test_out['Predicted'] = y_test_pred - 1
    df_test_out['Sample'] = 'Test'

    predictions_df = pd.concat([df_train_out, df_test_out])
    
    # Join with the original X to keep analysis columns
    all_data_for_analysis = pd.concat([X_train, X_test])
    predictions_df = predictions_df.join(all_data_for_analysis[['Open', 'Close', 'DVT_STD']], how='inner')
    
    pred_filename = f'data/{segment_name}_nifty50_rf_predictions.csv'
    predictions_df.to_csv(pred_filename)
    print(f"Predictions saved to '{pred_filename}'")

    return best_model, predictions_df

def model_training_pipeline():
    """Main pipeline to load data, segment it, and train models for each segment."""
    print("\n--- Starting Step 4 (v2): RandomForest Model Training ---")
    os.makedirs('models', exist_ok=True)

    # Load Final Dataset
    try:
        df_final = pd.read_csv('data/nifty50_final_dataset.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'nifty50_final_dataset.csv' not found. Please run feature_engineering.py first.")
        return
        
    # Prepare Data Segments
    df_mean_reverting = df_final[df_final['Segment'] == 'Mean Reverting'].copy()
    df_trending = df_final[df_final['Segment'] == 'Trending'].copy()

    df_mean_reverting.drop('Segment', axis=1, inplace=True)
    df_trending.drop('Segment', axis=1, inplace=True)
    
    # Split into X and y for each segment
    X_mr, y_mr = split_data_xy(df_mean_reverting)
    X_t, y_t = split_data_xy(df_trending)

    # Train Models for Each Segment
    model_mr, preds_mr = train_random_forest_model(X_mr, y_mr, 'MeanReverting')
    model_t, preds_t = train_random_forest_model(X_t, y_t, 'Trending')
    
    # Combine predictions for overall analysis
    all_predictions = pd.concat([preds_mr, preds_t]).sort_index()
    all_predictions.to_csv('data/nifty50_combined_rf_predictions.csv')
    print("\nCombined RandomForest predictions saved to 'data/nifty50_combined_rf_predictions.csv'")

    print("\n--- Step 4 (v2) Complete ---")

if __name__ == "__main__":
    model_training_pipeline()
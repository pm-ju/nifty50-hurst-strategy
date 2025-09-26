import os
from src import data_collection, hurst_calculator, feature_engineering, model_training, visualize_results

def run_full_pipeline():
    """
    Executes the entire NIFTY 50 Hurst exponent trading strategy pipeline.
    This main script controls the flow from data download to final performance visualization.
    """
    print("==========================================================")
    print("= Starting NIFTY 50 Hurst Exponent Strategy Pipeline     =")
    print("==========================================================")
    
    # --- Step 1: Data Collection ---
    # This function downloads the latest NIFTY 50 data and saves it.
    print("\n>>> STEP 1: Kicking off Data Collection...")
    try:
        data_collection.download_and_analyze_nifty50_data()
        print(">>> STEP 1: Data Collection Complete.\n")
    except Exception as e:
        print(f"!!! ERROR in Step 1 (Data Collection): {e}")
        return # Stop the pipeline if this fails

    # --- Step 2: Hurst Exponent Calculation ---
    # This calculates the rolling Hurst exponent and creates an analysis plot.
    print("\n>>> STEP 2: Kicking off Hurst Exponent Calculation...")
    try:
        hurst_calculator.create_hurst_analysis_for_nifty50()
        print(">>> STEP 2: Hurst Exponent Calculation Complete.\n")
    except Exception as e:
        print(f"!!! ERROR in Step 2 (Hurst Calculation): {e}")
        return

    # --- Step 3: Feature Engineering ---
    # This creates the market segments, target variable, and technical features.
    print("\n>>> STEP 3: Kicking off Feature Engineering...")
    try:
        feature_engineering.create_features_and_target_for_nifty50()
        print(">>> STEP 3: Feature Engineering Complete.\n")
    except Exception as e:
        print(f"!!! ERROR in Step 3 (Feature Engineering): {e}")
        return

    # --- Step 4: Model Training ---
    # This trains the RandomForest models for each market segment.
    print("\n>>> STEP 4: Kicking off Model Training (RandomForest)...")
    try:
        model_training.model_training_pipeline()
        print(">>> STEP 4: Model Training Complete.\n")
    except Exception as e:
        print(f"!!! ERROR in Step 4 (Model Training): {e}")
        return

    # --- Step 5: Performance Visualization ---
    # This runs the backtest and generates the final performance plots and metrics.
    print("\n>>> STEP 5: Kicking off Performance Visualization...")
    try:
        visualize_results.visualize_performance_pipeline()
        print(">>> STEP 5: Performance Visualization Complete.\n")
    except Exception as e:
        print(f"!!! ERROR in Step 5 (Visualization): {e}")
        return

    print("==========================================================")
    print("=                  PIPELINE COMPLETE                     =")
    print("= Check the /data, /models, and /results folders for output. =")
    print("==========================================================")


if __name__ == '__main__':
    # Set the working directory to the project's root folder for consistency
    # This ensures that file paths work correctly regardless of where you run the script from
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    run_full_pipeline()
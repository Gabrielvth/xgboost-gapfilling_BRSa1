# analyze_ameriflux_complete.py - Enhanced diagnostics

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ameriflux_transform import transform_ameriflux_data
from synthetic_data import make_synth_data
import CO2_gapfill
import random
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

def preprocess_data(data):
    """Apply outlier filtering and QC checks"""
    
    print("Original data shape:", data.shape)
    print(f"NEE range: {data['NEE'].min():.2f} to {data['NEE'].max():.2f}")
    
    # 1. Remove extreme outliers (|NEE| > 30 μmol m-2 s-1)
    outlier_mask = np.abs(data['NEE']) > 30
    print(f"Removing {outlier_mask.sum()} outliers where |NEE| > 30")
    
    # 2. Filter data
    data_filtered = data[~outlier_mask].copy()
    
    print("Filtered data shape:", data_filtered.shape)
    print(f"Filtered NEE range: {data_filtered['NEE'].min():.2f} to {data_filtered['NEE'].max():.2f}")
    
    return data_filtered

def create_synthetic_ameriflux_data():
    """Create synthetic datasets with proper preprocessing"""
    
    os.makedirs('Synthetic_AmeriFlux/', exist_ok=True)
    
    # Load and transform original data
    original_file = 'AMF_BR-Sa1_BASE_HH_6-5 copy.csv'
    data = transform_ameriflux_data(original_file)
    data = CO2_gapfill.add_time_vars(data)
    
    # Apply preprocessing (outlier filtering)
    data = preprocess_data(data)
    
    # Create multiple synthetic versions using existing function
    x_cols = ['PPFD_IN', 'TA_1_2_3', 'VPD_PI_1_2_1']
    y_col = 'NEE'
    
    for i in range(1, 6):
        print(f"Creating synthetic dataset {i}...")
        
        # Use the existing make_synth_data function
        synth_data = make_synth_data(data, x_cols, y_col)
        
        # REMOVE unit conversion - keep in original μmol units
        # synth_data['NEE'] = synth_data['NEE'] * 0.04401  # REMOVED
        
        synth_data.to_csv(f'Synthetic_AmeriFlux/BR-Sa1_synth_{i}.csv')
        print(f"Created synthetic dataset {i}")

def create_artificial_gaps():
    """Create artificial gap scenarios for 30%, 50%, 70% data coverage"""
    
    os.makedirs('Gap_Scenarios_AmeriFlux/', exist_ok=True)
    
    # Load one synthetic dataset to get time structure
    sample_data = pd.read_csv('Synthetic_AmeriFlux/BR-Sa1_synth_1.csv', index_col=0)
    
    for gap_percentage in [30, 50, 70]:
        gap_scenarios = create_gap_scenarios(sample_data, gap_percentage)
        gap_scenarios.to_csv(f'Gap_Scenarios_AmeriFlux/gaps_{gap_percentage}.csv')
        print(f"Created gap scenarios for {gap_percentage}% coverage")

def create_gap_scenarios(data, target_coverage):
    """Create random gap patterns to achieve target data coverage"""
    
    n_scenarios = 10
    gap_frame = pd.DataFrame(index=data.index)
    
    for scenario in range(n_scenarios):
        gaps = np.zeros(len(data))
        current_coverage = 100
        
        while current_coverage > target_coverage:
            # Create random gaps of various lengths (1-144 points = 3 days)
            gap_start = random.randint(0, len(data) - 144)
            gap_length = random.randint(1, 144)
            gaps[gap_start:gap_start + gap_length] = 1
            current_coverage = (1 - np.sum(gaps) / len(gaps)) * 100
        
        gap_frame[f'scenario_{scenario}'] = gaps
    
    return gap_frame

def run_gapfilling_experiment():
    """Run complete gapfilling experiment with corrected calculations"""
    
    os.makedirs('Results_AmeriFlux_Complete/', exist_ok=True)
    results = []
    
    x_cols = ['PPFD_IN', 'TA_1_2_3', 'VPD_PI_1_2_1', 
              'Month_sin', 'Month_cos', 'Hour_sin', 'Hour_cos', 'Time']
    y_col = 'NEE'
    
    # Process each synthetic dataset
    for dataset_num in range(1, 6):
        print(f"Processing synthetic dataset {dataset_num}")
        
        # Load synthetic data
        synth_file = f'Synthetic_AmeriFlux/BR-Sa1_synth_{dataset_num}.csv'
        orig_data = pd.read_csv(synth_file, index_col=0)
        orig_data.index = pd.to_datetime(orig_data.index)
        orig_data = CO2_gapfill.add_time_vars(orig_data)
        
        # Apply same preprocessing
        orig_data = preprocess_data(orig_data)
        
        # Optimize hyperparameters
        hyperparams = CO2_gapfill.optimize_hyperparameters(orig_data, x_cols, y_col)
        
        # Test different gap scenarios
        for gap_perc in [30, 50, 70]:
            gap_file = pd.read_csv(f'Gap_Scenarios_AmeriFlux/gaps_{gap_perc}.csv', index_col=0)
            
            for scenario in range(3):  # Reduce to 3 scenarios for faster testing
                print(f"  Gap {gap_perc}%, scenario {scenario}")
                
                # Create dataset with artificial gaps
                data = orig_data.copy()
                data['complete_NEE'] = data['NEE'].copy()
                
                # Insert artificial gaps
                if scenario < len(gap_file.columns):
                    gap_mask = gap_file.iloc[:len(data), scenario].values == 1
                    data.loc[gap_mask, 'NEE'] = np.nan
                
                # Perform gapfilling
                gapfilled_data = CO2_gapfill.cv_preds(data, x_cols, y_col, hyperparams, 10)
                
                # Calculate error metrics
                error_metrics = calculate_comprehensive_errors(data, gapfilled_data)
                error_metrics.update({
                    'Dataset': dataset_num,
                    'Gap_Percentage': gap_perc,
                    'Scenario': scenario
                })
                results.append(error_metrics)
    
    # Save comprehensive results
    results_df = pd.DataFrame(results)
    results_df.to_csv('Results_AmeriFlux_Complete/comprehensive_results.csv', index=False)
    
    # Print diagnostic summary
    print("\n=== DIAGNOSTIC SUMMARY ===")
    print("1. Unit conversions applied:")
    print("   - Removed μmol to mg conversion (keeping original units)")
    print("   - Balance Error: μmol m-2 s-1 → g C m-2 yr-1 using conversion factor")
    print("   - RMSE: kept in μmol m-2 s-1")
    
    print("\n2. Data filters applied:")
    print("   - Outlier removal: |NEE| > 30 μmol m-2 s-1")
    
    print("\n3. RMSE calculation:")
    print("   - Computed ONLY on gap timestamps")
    print("   - Excluded any remaining NaN values")
    
    # Summary statistics
    summary = results_df.groupby('Gap_Percentage')[['Balance_Error', 'RMSE', 'R2']].agg(['mean', 'std'])
    print("\n4. Corrected Results:")
    print("Gap Coverage | Balance Error (g C m-2 yr-1) | RMSE (μmol m-2 s-1) | R²")
    print("-" * 70)
    for gap_perc in [30, 50, 70]:
        subset = results_df[results_df['Gap_Percentage'] == gap_perc]
        be_mean = subset['Balance_Error'].mean()
        be_std = subset['Balance_Error'].std()
        rmse_mean = subset['RMSE'].mean()
        rmse_std = subset['RMSE'].std()
        r2_mean = subset['R2'].mean()
        print(f"{gap_perc:3d}%        | {be_mean:8.2f} ± {be_std:5.2f}        | {rmse_mean:6.2f} ± {rmse_std:4.2f}      | {r2_mean:.3f}")
    
    return results_df

def calculate_comprehensive_errors(original_data, gapfilled_data):
    """Calculate error metrics with proper unit handling"""
    
    # Get gap mask (where artificial gaps were inserted)
    gap_mask = original_data['NEE'].isnull()
    n_gaps = gap_mask.sum()
    
    if n_gaps == 0:
        return {
            'Balance_Error': np.nan,
            'Bias': np.nan,
            'RMSE': np.nan,
            'R2': np.nan,
            'Gap_Count': 0,
            'Total_Points': len(original_data)
        }
    
    # 1. RMSE calculation - ONLY on gap timestamps
    gap_original = original_data.loc[gap_mask, 'complete_NEE']
    gap_predicted = gapfilled_data.loc[gap_mask, 'modelled_NEE']
    
    # Remove any remaining NaNs
    valid_mask = ~(pd.isna(gap_original) | pd.isna(gap_predicted))
    gap_original_clean = gap_original[valid_mask]
    gap_predicted_clean = gap_predicted[valid_mask]
    
    if len(gap_original_clean) > 0:
        # RMSE in μmol m-2 s-1 (original units)
        rmse = np.sqrt(np.mean((gap_predicted_clean - gap_original_clean) ** 2))
        bias = np.mean(gap_predicted_clean - gap_original_clean)
        r2 = np.corrcoef(gap_original_clean, gap_predicted_clean)[0, 1] ** 2
    else:
        rmse = bias = r2 = np.nan
    
    # 2. Annual balance error
    # Convert from μmol m-2 s-1 to g C m-2 yr-1
    # 1 μmol C = 12 μg C = 0.012 mg C = 0.000012 g C
    # 30-min data points per year ≈ 17520
    # Conversion: μmol m-2 s-1 * 0.000012 g/μmol * 86400 s/day * 365.25 days/yr
    conversion_factor = 0.000012 * 86400 * 365.25  # ≈ 0.378
    
    # For 30-minute data: sum * 1800 seconds * conversion
    original_balance = original_data['complete_NEE'].sum() * 1800 * 0.000012 * 365.25 / 365.25  # g C m-2 yr-1
    
    # Use gapfilled NEE for the final balance
    gapfilled_balance = gapfilled_data['modelled_NEE'].fillna(0).sum() * 1800 * 0.000012 * 365.25 / 365.25
    
    # Simpler approach: use the ratio method from original paper
    # Annual balance = sum of 30-min fluxes * 1800 * conversion factor
    balance_error = (gapfilled_balance - original_balance)
    
    print(f"Debug - Gap count: {n_gaps}, RMSE: {rmse:.2f}, Balance error: {balance_error:.2f}")
    
    return {
        'Balance_Error': balance_error,  # g C m-2 yr-1
        'Bias': bias,                    # μmol m-2 s-1
        'RMSE': rmse,                    # μmol m-2 s-1
        'R2': r2,
        'Gap_Count': n_gaps,
        'Total_Points': len(original_data)
    }

def create_comprehensive_plots():
    """Create comprehensive plots similar to original experiment"""
    
    results = pd.read_csv('Results_AmeriFlux_Complete/comprehensive_results.csv')
    
    # Plot 1: Balance error vs gap percentage (similar to plot_fig3)
    plt.figure(figsize=[12, 8])
    
    for gap_perc in [30, 50, 70]:
        data_subset = results[results['Gap_Percentage'] == gap_perc]
        plt.boxplot([data_subset['Balance_Error']], 
                   positions=[gap_perc], 
                   widths=8,
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
    
    plt.xlabel('Data Coverage (%)', fontsize=14)
    plt.ylabel('Balance Error [g C m$^{-2}$ y$^{-1}$]', fontsize=14)
    plt.title('XGBoost Gapfilling Performance - BR-Sa1', fontsize=16)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.savefig('Results_AmeriFlux_Complete/balance_error_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: RMSE comparison
    plt.figure(figsize=[10, 6])
    rmse_summary = results.groupby('Gap_Percentage')['RMSE'].agg(['mean', 'std'])
    
    plt.errorbar(rmse_summary.index, rmse_summary['mean'], 
                yerr=rmse_summary['std'], fmt='o-', capsize=5)
    plt.xlabel('Data Coverage (%)', fontsize=14)
    plt.ylabel('RMSE [μmol m$^{-2}$ s$^{-1}$]', fontsize=14)
    plt.title('RMSE vs Data Coverage', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig('Results_AmeriFlux_Complete/rmse_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_gap_distribution(data, gap_scenarios):
    """Task 1: Gap distribution analysis"""
    
    results = {}
    
    for gap_perc in [30, 50, 70]:
        gap_file = pd.read_csv(f'Gap_Scenarios_AmeriFlux/gaps_{gap_perc}.csv', index_col=0)
        
        # Average gap pattern across scenarios
        gap_pattern = gap_file.mean(axis=1)
        
        # Add datetime info
        gap_df = pd.DataFrame({'gap_prob': gap_pattern})
        gap_df.index = pd.to_datetime(data.index[:len(gap_df)])
        gap_df['month'] = gap_df.index.month
        gap_df['hour'] = gap_df.index.hour
        
        # Monthly gap distribution
        monthly_gaps = gap_df.groupby('month')['gap_prob'].mean()
        
        # Hourly gap distribution  
        hourly_gaps = gap_df.groupby('hour')['gap_prob'].mean()
        
        results[gap_perc] = {
            'monthly': monthly_gaps,
            'hourly': hourly_gaps,
            'seasonal_bias': monthly_gaps.std(),
            'diurnal_bias': hourly_gaps.std()
        }
        
        print(f"\nGap {gap_perc}% - Distribution Analysis:")
        print(f"Seasonal bias (std): {monthly_gaps.std():.3f}")
        print(f"Diurnal bias (std): {hourly_gaps.std():.3f}")
    
    return results

def analyze_daynight_balance(data, gap_mask):
    """Task 2: Day/night balance analysis"""
    
    # Define day/night based on PAR (using PPFD_IN)
    day_mask = data['PPFD_IN'] >= 10  # µmol m-2 s-1
    night_mask = data['PPFD_IN'] < 10
    
    # Training set (non-gaps)
    training_mask = ~gap_mask
    train_day_frac = (training_mask & day_mask).sum() / training_mask.sum()
    train_night_frac = (training_mask & night_mask).sum() / training_mask.sum()
    
    # Gap set (validation)
    gap_day_frac = (gap_mask & day_mask).sum() / gap_mask.sum()
    gap_night_frac = (gap_mask & night_mask).sum() / gap_mask.sum()
    
    imbalance = abs(train_night_frac - gap_night_frac) * 100
    
    print(f"\nDay/Night Balance Analysis:")
    print(f"Training set - Day: {train_day_frac:.3f}, Night: {train_night_frac:.3f}")
    print(f"Gap set - Day: {gap_day_frac:.3f}, Night: {gap_night_frac:.3f}")
    print(f"Night fraction imbalance: {imbalance:.1f} percentage points")
    
    return {
        'train_night_frac': train_night_frac,
        'gap_night_frac': gap_night_frac,
        'imbalance': imbalance,
        'needs_rebalancing': imbalance > 10
    }

def hyperparameter_tuning_50pct(data, x_cols, y_col):
    """Task 3: Enhanced hyperparameter tuning for 50% scenario"""
    
    # Prepare data
    X = data[x_cols].dropna()
    y = data[y_col].dropna()
    
    # Align data
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    # Create month groups for GroupKFold
    groups = pd.to_datetime(X.index).month
    
    # Enhanced parameter grid
    param_grid = {
        'n_estimators': [300, 500, 800, 1200],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    # Randomized search with GroupKFold
    xgb_model = xgb.XGBRegressor(random_state=42)
    group_kfold = GroupKFold(n_splits=3)
    
    random_search = RandomizedSearchCV(
        xgb_model, 
        param_grid, 
        n_iter=50,
        cv=group_kfold,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X, y, groups=groups)
    
    print(f"\nOptimal hyperparameters for 50% scenario:")
    print(random_search.best_params_)
    
    return random_search.best_params_

def test_outlier_sensitivity(data, thresholds=[30, 40]):
    """Task 4: Test different outlier filtering thresholds"""
    
    results = {}
    
    for threshold in thresholds:
        # Apply filtering
        outlier_mask = np.abs(data['NEE']) > threshold
        filtered_data = data[~outlier_mask].copy()
        
        print(f"\nOutlier threshold {threshold}: removed {outlier_mask.sum()} points ({outlier_mask.mean()*100:.1f}%)")
        
        results[threshold] = {
            'data_shape': filtered_data.shape,
            'removed_points': outlier_mask.sum(),
            'removed_fraction': outlier_mask.mean()
        }
    
    return results

def seasonal_block_cv(data, x_cols, y_col):
    """Task 5: Seasonal block cross-validation"""
    
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5], 
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }
    
    results = []
    
    for test_season, months in seasons.items():
        # Split data
        test_mask = pd.to_datetime(data.index).month.isin(months)
        train_data = data[~test_mask].copy()
        test_data = data[test_mask].copy()
        
        # Add artificial gaps to test data
        gap_mask = np.random.choice([True, False], size=len(test_data), p=[0.5, 0.5])
        test_data.loc[gap_mask, 'NEE'] = np.nan
        test_data['complete_NEE'] = data.loc[test_mask, 'NEE'].copy()
        
        # Train and predict
        hyperparams = CO2_gapfill.optimize_hyperparameters(train_data, x_cols, y_col)
        gapfilled = CO2_gapfill.cv_preds(test_data, x_cols, y_col, hyperparams, 5)
        
        # Calculate metrics
        gap_original = test_data.loc[gap_mask, 'complete_NEE']
        gap_predicted = gapfilled.loc[gap_mask, 'modelled_NEE']
        
        # Remove NaNs
        valid_mask = ~(pd.isna(gap_original) | pd.isna(gap_predicted))
        if valid_mask.sum() > 0:
            rmse = np.sqrt(np.mean((gap_predicted[valid_mask] - gap_original[valid_mask]) ** 2))
            
            # Balance error
            conversion = 0.5 * 8766 * 12.01 * 1e-6
            be = abs(gap_predicted.sum() - gap_original.sum()) * conversion
        else:
            rmse = be = np.nan
            
        results.append({
            'season': test_season,
            'rmse': rmse,
            'balance_error': be,
            'n_gaps': gap_mask.sum()
        })
        
        print(f"Season {test_season}: RMSE={rmse:.2f}, BE={be:.2f}")
    
    return pd.DataFrame(results)

def comprehensive_diagnostic():
    """Run all diagnostic tasks"""
    
    print("=== COMPREHENSIVE DIAGNOSTIC FOR 50% GAP BIAS ===\n")
    
    # Load data
    original_file = 'AMF_BR-Sa1_BASE_HH_6-5 copy.csv'
    data = transform_ameriflux_data(original_file)
    data = CO2_gapfill.add_time_vars(data)
    data = preprocess_data(data)
    
    # Task 1: Gap distribution analysis
    print("TASK 1: Gap Distribution Analysis")
    gap_analysis = analyze_gap_distribution(data, None)
    
    # Task 2: Day/night balance (for 50% scenario)
    print("\nTASK 2: Day/Night Balance Analysis")
    gap_file_50 = pd.read_csv('Gap_Scenarios_AmeriFlux/gaps_50.csv', index_col=0)
    gap_mask_50 = gap_file_50.iloc[:len(data), 0].values == 1
    balance_analysis = analyze_daynight_balance(data, gap_mask_50)
    
    # Task 3: Hyperparameter tuning
    print("\nTASK 3: Enhanced Hyperparameter Tuning")
    x_cols = ['PPFD_IN', 'TA_1_2_3', 'VPD_PI_1_2_1', 
              'Month_sin', 'Month_cos', 'Hour_sin', 'Hour_cos', 'Time']
    optimal_params = hyperparameter_tuning_50pct(data, x_cols, 'NEE')
    
    # Task 4: Outlier sensitivity
    print("\nTASK 4: Outlier Filter Sensitivity")
    outlier_analysis = test_outlier_sensitivity(data)
    
    # Task 5: Seasonal CV
    print("\nTASK 5: Seasonal Block Cross-Validation")
    seasonal_results = seasonal_block_cv(data, x_cols, 'NEE')
    
    # Summary
    print("\n=== SUMMARY ===")
    print("Factors most likely to reduce 50% BE bias:")
    if balance_analysis['needs_rebalancing']:
        print("1. Day/night imbalance detected - implement stratified sampling")
    if gap_analysis[50]['seasonal_bias'] > 0.1:
        print("2. Seasonal gap bias detected - consider seasonal stratification")
    if gap_analysis[50]['diurnal_bias'] > 0.1:
        print("3. Diurnal gap bias detected - add time-of-day features")
    
    return {
        'gap_analysis': gap_analysis,
        'balance_analysis': balance_analysis,
        'optimal_params': optimal_params,
        'outlier_analysis': outlier_analysis,
        'seasonal_results': seasonal_results
    }

def main():
    """Run complete AmeriFlux gapfilling evaluation"""
    
    print("Step 1: Creating synthetic datasets...")
    create_synthetic_ameriflux_data()
    
    print("Step 2: Creating artificial gap scenarios...")
    create_artificial_gaps()
    
    print("Step 3: Running gapfilling experiments...")
    results_df = run_gapfilling_experiment()
    
    print("Step 4: Creating comprehensive plots...")
    create_comprehensive_plots()
    
    # Run comprehensive diagnostics
    diagnostic_results = comprehensive_diagnostic()
    
    # Save results
    pd.DataFrame(diagnostic_results['seasonal_results']).to_csv(
        'Results_AmeriFlux_Complete/seasonal_cv_results.csv', index=False)
    
    print("\nExperiment complete!")
    print("Results summary:")
    print(results_df.groupby('Gap_Percentage')[['Balance_Error', 'RMSE', 'R2']].agg(['mean', 'std']))

if __name__ == "__main__":
    main()
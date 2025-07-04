# XGBoost Gap-Filling Analysis for AmeriFlux Data

This repository contains code to evaluate XGBoost gap-filling performance on AmeriFlux eddy covariance data for the BR-Sa1 site in the Brazilian Amazon.

## Overview

This study replicates and extends the methodology from Vekuri et al. (2023) to assess systematic bias in carbon balance estimates from XGBoost gap-filling of CO2 flux data.

## Files

- analyze_ameriflux_complete.py - Main experiment framework
- final_analysis_documentation.py - Final analysis and documentation  
- ameriflux_transform.py - Data transformation utilities
- CO2_gapfill.py - XGBoost gap-filling implementation
- synthetic_data.py - Synthetic data generation
- requirements.txt - Python dependencies

## Installation

Clone this repository and install dependencies:

    git clone https://github.com/yourusername/ameriflux-xgboost-gapfilling.git
    cd ameriflux-xgboost-gapfilling
    pip install -r requirements.txt

Download BR-Sa1 data from AmeriFlux and place the CSV file in the project directory.

## Usage

Run the main analysis:

    python final_analysis_documentation.py

## Key Results

- RMSE approximately 4.6 micromol per square meter per second
- Balance errors within target range for moderate gap scenarios
- R-squared values around 0.77

## Citation

If you use this code, please cite the original methodology by Vekuri et al. (2023).

## License

MIT License
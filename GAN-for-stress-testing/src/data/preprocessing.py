"""
Data processing utilities for GAN-based stress testing.

This module handles data loading, preprocessing, and feature engineering
for both borrower-level and macroeconomic data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import Dict, Tuple, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FeatureConfig:
    """Configuration for borrower and macro features."""
    
    # Borrower-level features
    BORROWER_FEATURES = [
        'fico_score',           # Credit bureau score
        'dti_ratio',            # Debt-to-income ratio
        'ltv_ratio',            # Loan-to-value ratio
        'income_annual',        # Annual income
        'employment_length',    # Employment length in years
        'delinq_2yrs',         # Number of delinquencies in past 2 years
        'credit_util_ratio',    # Credit utilization ratio
        'open_accounts',        # Number of open credit accounts
        'total_accounts',       # Total number of credit accounts
        'payment_history_score',# Payment history score (derived)
        'loan_amount',          # Current loan amount
        'interest_rate',        # Interest rate on loan
        'loan_term',            # Loan term in months
        'property_value',       # Property value (for mortgages)
        'age'                   # Borrower age
    ]
    
    # Macroeconomic features
    MACRO_FEATURES = [
        'unemployment_rate',    # Unemployment rate (%)
        'gdp_growth_rate',      # GDP growth rate (%)
        'interest_rate_fed',    # Federal interest rate (%)
        'inflation_rate',       # Inflation rate (%)
        'housing_price_index',  # Housing price index
        'market_volatility'     # Market volatility index (VIX-like)
    ]
    
    # Target variable
    TARGET = 'default_flag'  # 1 if default, 0 otherwise


class DataProcessor:
    """
    Main data processing class for stress testing pipeline.
    
    Handles loading, cleaning, and preprocessing of borrower and macro data.
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.borrower_scaler = StandardScaler()
        self.macro_scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        
    def load_data(self, 
                  borrower_file: str, 
                  macro_file: str,
                  date_column: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load borrower and macroeconomic data.
        
        Args:
            borrower_file: Path to borrower data CSV
            macro_file: Path to macro data CSV
            date_column: Name of date column for merging
            
        Returns:
            Tuple of (borrower_df, macro_df)
        """
        logger.info(f"Loading borrower data from {borrower_file}")
        borrower_df = pd.read_csv(borrower_file)
        
        logger.info(f"Loading macro data from {macro_file}")
        macro_df = pd.read_csv(macro_file)
        
        # Parse dates
        borrower_df[date_column] = pd.to_datetime(borrower_df[date_column])
        macro_df[date_column] = pd.to_datetime(macro_df[date_column])
        
        logger.info(f"Loaded {len(borrower_df)} borrower records and {len(macro_df)} macro records")
        
        return borrower_df, macro_df
    
    def merge_data(self, 
                   borrower_df: pd.DataFrame, 
                   macro_df: pd.DataFrame,
                   date_column: str = 'date') -> pd.DataFrame:
        """
        Merge borrower and macro data on date.
        
        Args:
            borrower_df: Borrower dataframe
            macro_df: Macro dataframe  
            date_column: Date column for merging
            
        Returns:
            Merged dataframe
        """
        # Merge on date (using left join to keep all borrower records)
        merged_df = borrower_df.merge(
            macro_df, 
            on=date_column, 
            how='left'
        )
        
        logger.info(f"Merged data shape: {merged_df.shape}")
        
        return merged_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        initial_rows = len(df)
        
        # Remove rows with missing target
        if 'default_flag' in df.columns:
            df = df.dropna(subset=['default_flag'])
            
        # Handle missing values in features
        # For borrower features - use median imputation
        for feature in FeatureConfig.BORROWER_FEATURES:
            if feature in df.columns:
                df[feature] = df[feature].fillna(df[feature].median())
                
        # For macro features - forward fill (time series nature)
        for feature in FeatureConfig.MACRO_FEATURES:
            if feature in df.columns:
                df[feature] = df[feature].fillna(method='ffill')
        
        # Remove any remaining rows with NaN
        df = df.dropna()
        
        final_rows = len(df)
        logger.info(f"Data cleaning: {initial_rows} -> {final_rows} rows ({initial_rows-final_rows} removed)")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with engineered features
        """
        df = df.copy()
        
        # Borrower feature engineering
        if 'fico_score' in df.columns:
            # FICO score bins
            df['fico_high'] = (df['fico_score'] >= 740).astype(int)
            df['fico_low'] = (df['fico_score'] < 600).astype(int)
            
        if 'dti_ratio' in df.columns:
            # High DTI flag
            df['high_dti'] = (df['dti_ratio'] > 0.36).astype(int)
            
        if 'ltv_ratio' in df.columns:
            # High LTV flag  
            df['high_ltv'] = (df['ltv_ratio'] > 0.8).astype(int)
            
        # Payment history score (synthetic if not available)
        if 'payment_history_score' not in df.columns:
            if 'delinq_2yrs' in df.columns and 'fico_score' in df.columns:
                df['payment_history_score'] = (
                    0.7 * df['fico_score'] / 850 + 
                    0.3 * (1 - np.clip(df['delinq_2yrs'] / 5, 0, 1))
                ) * 100
        
        # Macro feature interactions
        if 'unemployment_rate' in df.columns and 'gdp_growth_rate' in df.columns:
            df['econ_stress_index'] = df['unemployment_rate'] - df['gdp_growth_rate']
            
        logger.info(f"Feature engineering completed. Shape: {df.shape}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features for model training/inference.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (borrower_features, macro_features, targets)
        """
        # Extract borrower features
        borrower_cols = [col for col in FeatureConfig.BORROWER_FEATURES if col in df.columns]
        borrower_features = df[borrower_cols].values
        
        # Extract macro features
        macro_cols = [col for col in FeatureConfig.MACRO_FEATURES if col in df.columns]
        macro_features = df[macro_cols].values
        
        # Extract targets
        if 'default_flag' in df.columns:
            targets = df['default_flag'].values
        else:
            targets = None
            
        logger.info(f"Prepared features - Borrower: {borrower_features.shape}, Macro: {macro_features.shape}")
        
        return borrower_features, macro_features, targets
    
    def fit_scalers(self, borrower_features: np.ndarray, macro_features: np.ndarray):
        """
        Fit scalers on training data.
        
        Args:
            borrower_features: Borrower feature array
            macro_features: Macro feature array
        """
        self.borrower_scaler.fit(borrower_features)
        self.macro_scaler.fit(macro_features)
        self.is_fitted = True
        
        logger.info("Scalers fitted successfully")
    
    def transform_features(self, 
                          borrower_features: np.ndarray, 
                          macro_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform features using fitted scalers.
        
        Args:
            borrower_features: Borrower feature array
            macro_features: Macro feature array
            
        Returns:
            Tuple of transformed (borrower_features, macro_features)
        """
        if not self.is_fitted:
            raise ValueError("Scalers must be fitted before transformation")
            
        borrower_scaled = self.borrower_scaler.transform(borrower_features)
        macro_scaled = self.macro_scaler.transform(macro_features)
        
        return borrower_scaled, macro_scaled
    
    def create_stress_scenarios(self) -> Dict[str, np.ndarray]:
        """
        Create predefined stress scenarios for testing.
        
        Returns:
            Dictionary of stress scenarios
        """
        scenarios = {
            'baseline': np.array([
                [5.0, 2.0, 2.5, 2.0, 100.0, 15.0]  # Normal economic conditions
            ]),
            'mild_recession': np.array([
                [7.5, -1.0, 1.5, 3.0, 95.0, 25.0]  # Mild recession
            ]),
            'severe_recession': np.array([
                [10.0, -3.5, 0.5, 4.0, 85.0, 40.0]  # Severe recession
            ]),
            'housing_crisis': np.array([
                [8.0, -2.0, 0.25, 2.5, 70.0, 35.0]  # Housing market crash
            ]),
            'inflation_crisis': np.array([
                [6.0, 1.0, 5.0, 8.0, 105.0, 30.0]  # High inflation scenario
            ])
        }
        
        # Transform scenarios using fitted scaler
        if self.is_fitted:
            for name, scenario in scenarios.items():
                scenarios[name] = self.macro_scaler.transform(scenario)
        
        return scenarios

    @staticmethod
    def create_synthetic_data(n_samples: int = 10000, 
                             random_state: int = 42) -> pd.DataFrame:
        """
        Create synthetic dataset for testing purposes.

        Args:
            n_samples: Number of samples to generate
            random_state: Random seed

        Returns:
            Synthetic dataframe
        """
        np.random.seed(random_state)

        # Generate borrower features
        data = {
            'fico_score': np.random.normal(720, 80, n_samples).clip(300, 850),
            'dti_ratio': np.random.beta(2, 5, n_samples) * 0.6,  # Most people have reasonable DTI
            'ltv_ratio': np.random.beta(3, 2, n_samples) * 1.2,  # Some people have high LTV
            'income_annual': np.random.lognormal(11, 0.5, n_samples),  # Log-normal income distribution
            'employment_length': np.random.exponential(5, n_samples).clip(0, 30),
            'delinq_2yrs': np.random.poisson(0.5, n_samples).clip(0, 10),
            'credit_util_ratio': np.random.beta(2, 3, n_samples),
            'open_accounts': np.random.poisson(8, n_samples).clip(1, 30),
            'total_accounts': np.random.poisson(12, n_samples).clip(1, 50),
            'loan_amount': np.random.lognormal(12, 0.6, n_samples),
            'interest_rate': np.random.normal(6, 2, n_samples).clip(1, 30),
            'loan_term': np.random.choice([180, 240, 360], n_samples, p=[0.1, 0.3, 0.6]),
            'property_value': np.random.lognormal(13, 0.4, n_samples),
            'age': np.random.normal(40, 12, n_samples).clip(18, 80)
        }

        # Generate macro features (time series)
        dates = pd.date_range('2010-01-01', periods=n_samples, freq='D')

        # Simple AR(1) processes for macro variables
        unemployment = 5.0 + np.cumsum(np.random.normal(0, 0.1, n_samples)) * 0.1
        gdp_growth = 2.0 + np.cumsum(np.random.normal(0, 0.1, n_samples)) * 0.05

        data.update({
            'date': dates,
            'unemployment_rate': unemployment.clip(1, 15),
            'gdp_growth_rate': gdp_growth.clip(-5, 8),
            'interest_rate_fed': np.random.normal(3, 1, n_samples).clip(0, 10),
            'inflation_rate': np.random.normal(2.5, 1, n_samples).clip(-2, 10),
            'housing_price_index': 100 + np.cumsum(np.random.normal(0.01, 0.02, n_samples)),
            'market_volatility': np.random.gamma(2, 10, n_samples)
        })

        df = pd.DataFrame(data)

        # Generate target variable (default) based on features
        # Higher default probability with worse borrower characteristics and macro conditions
        logit = (
            -3.0 +  # Base probability
            -0.01 * (df['fico_score'] - 600) +  # FICO effect
            2.0 * df['dti_ratio'] +  # DTI effect
            1.5 * df['ltv_ratio'] +  # LTV effect
            0.5 * df['delinq_2yrs'] +  # Delinquency effect
            0.1 * df['unemployment_rate'] +  # Unemployment effect
            -0.2 * df['gdp_growth_rate']  # GDP effect
        )

        default_prob = 1 / (1 + np.exp(-logit))
        data['default_flag'] = np.random.binomial(1, default_prob, n_samples)

        df['default_flag'] = data['default_flag']

        logger.info(f"Created synthetic dataset with {n_samples} samples")
        logger.info(f"Default rate: {df['default_flag'].mean():.3f}")

        return df

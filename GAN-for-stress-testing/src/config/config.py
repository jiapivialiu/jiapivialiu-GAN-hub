"""
Configuration management for GAN-based stress testing.

This module handles configuration parameters and settings for the entire
stress testing pipeline.
"""

import yaml
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    # Data paths
    raw_data_path: str = "data/raw/"
    processed_data_path: str = "data/processed/"
    synthetic_data_path: str = "data/synthetic/"
    
    # Feature configuration
    numerical_features: List[str] = None
    categorical_features: List[str] = None
    target_feature: str = "default_flag"
    
    # Data processing parameters
    train_test_split: float = 0.8
    validation_split: float = 0.2
    random_state: int = 42
    
    # Preprocessing options
    normalize_features: bool = True
    handle_missing: str = "median"  # 'mean', 'median', 'mode', 'drop'
    outlier_method: str = "iqr"  # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold: float = 3.0
    
    def __post_init__(self):
        if self.numerical_features is None:
            self.numerical_features = [
                'age', 'income_annual', 'dti_ratio', 'fico_score',
                'loan_amount', 'employment_length', 'ltv_ratio'
            ]
        if self.categorical_features is None:
            self.categorical_features = [
                'purpose', 'grade', 'emp_title_category'
            ]


@dataclass
class MacroConfig:
    """Configuration for macroeconomic scenarios."""
    # Scenario definitions
    baseline_scenario: Dict[str, float] = None
    stress_scenarios: Dict[str, Dict[str, float]] = None
    
    # Macro variables
    macro_variables: List[str] = None
    
    def __post_init__(self):
        if self.baseline_scenario is None:
            self.baseline_scenario = {
                'gdp_growth': 2.0,
                'unemployment_rate': 5.0,
                'interest_rate': 3.0,
                'inflation_rate': 2.5,
                'house_price_index': 100.0
            }
        
        if self.stress_scenarios is None:
            self.stress_scenarios = {
                'adverse': {
                    'gdp_growth': -2.0,
                    'unemployment_rate': 8.5,
                    'interest_rate': 1.0,
                    'inflation_rate': 1.0,
                    'house_price_index': 85.0
                },
                'severely_adverse': {
                    'gdp_growth': -4.0,
                    'unemployment_rate': 12.0,
                    'interest_rate': 0.5,
                    'inflation_rate': 0.5,
                    'house_price_index': 70.0
                }
            }
        
        if self.macro_variables is None:
            self.macro_variables = list(self.baseline_scenario.keys())


@dataclass
class GANConfig:
    """Configuration for Conditional GAN."""
    # Architecture parameters
    generator_layers: List[int] = None
    discriminator_layers: List[int] = None
    latent_dim: int = 100
    condition_dim: int = 5  # Number of macro variables
    
    # Training parameters
    batch_size: int = 256
    num_epochs: int = 1000
    learning_rate_generator: float = 0.0003      # Increased for better generator training
    learning_rate_discriminator: float = 0.0001  # Reduced to prevent discriminator dominance
    beta1: float = 0.5
    beta2: float = 0.999
    
    # Training schedule
    discriminator_steps: int = 1
    generator_steps: int = 2  # Train generator more frequently to prevent collapse
    
    # Regularization
    dropout_rate: float = 0.3
    weight_decay: float = 1e-5
    gradient_penalty_weight: float = 50.0  # Increased for better stability
    
    # Early stopping
    patience: int = 50
    min_delta: float = 1e-4
    
    def __post_init__(self):
        if self.generator_layers is None:
            self.generator_layers = [256, 512, 512, 256]
        if self.discriminator_layers is None:
            self.discriminator_layers = [256, 512, 256, 128]


@dataclass
class PDModelConfig:
    """Configuration for PD models."""
    # Model types to train
    model_types: List[str] = None
    
    # Logistic regression parameters
    logistic_max_iter: int = 1000
    logistic_C: float = 1.0
    logistic_solver: str = 'liblinear'
    
    # Gradient boosting parameters
    gb_n_estimators: int = 100
    gb_max_depth: int = 6
    gb_learning_rate: float = 0.1
    gb_subsample: float = 0.8
    gb_random_state: int = 42
    
    # Cross-validation
    cv_folds: int = 5
    scoring_metric: str = 'roc_auc'
    
    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ['logistic', 'gradient_boosting', 'macro_conditioned']


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    # Metrics to compute
    compute_ks_test: bool = True
    compute_wasserstein: bool = True
    compute_correlation_preservation: bool = True
    compute_mode_collapse: bool = True
    
    # Stress testing metrics
    compute_pd_shift: bool = True
    compute_ecl_impact: bool = True
    compute_capital_impact: bool = True
    compute_scenario_discrimination: bool = True
    
    # ECL parameters
    lgd: float = 0.45  # Loss Given Default
    current_capital_ratio: float = 0.12
    
    # Thresholds
    ks_significance_level: float = 0.05
    mode_collapse_threshold: float = 0.1
    
    # Plotting
    save_plots: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    data: DataConfig
    macro: MacroConfig
    gan: GANConfig
    pd_model: PDModelConfig
    evaluation: EvaluationConfig
    
    # General settings
    project_name: str = "GAN_Stress_Testing"
    random_seed: int = 42
    log_level: str = "INFO"
    
    # Paths
    model_save_path: str = "models/"
    results_save_path: str = "results/"
    logs_path: str = "logs/"
    
    # Computing resources
    device: str = "cpu"  # 'auto', 'cpu', 'cuda'
    num_workers: int = 4
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            macro=MacroConfig(**config_dict.get('macro', {})),
            gan=GANConfig(**config_dict.get('gan', {})),
            pd_model=PDModelConfig(**config_dict.get('pd_model', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['data', 'macro', 'gan', 'pd_model', 'evaluation']}
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'Config':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return asdict(self)
    
    def save_yaml(self, yaml_path: Union[str, Path]):
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {yaml_path}")
    
    def save_json(self, json_path: Union[str, Path]):
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {json_path}")
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        config_dict = self.to_dict()
        deep_update(config_dict, updates)
        
        # Recreate config with updated values
        updated_config = self.from_dict(config_dict)
        
        # Update current instance
        self.data = updated_config.data
        self.macro = updated_config.macro
        self.gan = updated_config.gan
        self.pd_model = updated_config.pd_model
        self.evaluation = updated_config.evaluation
        
        # Update general settings
        for attr in ['project_name', 'random_seed', 'log_level', 'model_save_path', 
                     'results_save_path', 'logs_path', 'device', 'num_workers']:
            if hasattr(updated_config, attr):
                setattr(self, attr, getattr(updated_config, attr))


def get_default_config() -> Config:
    """Get default configuration."""
    return Config(
        data=DataConfig(),
        macro=MacroConfig(),
        gan=GANConfig(),
        pd_model=PDModelConfig(),
        evaluation=EvaluationConfig()
    )


def setup_logging(config: Config):
    """Setup logging based on configuration."""
    log_level = getattr(logging, config.log_level.upper())
    
    # Create logs directory if it doesn't exist
    Path(config.logs_path).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(config.logs_path) / 'stress_testing.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging setup complete. Log level: {config.log_level}")


def validate_config(config: Config) -> List[str]:
    """
    Validate configuration and return list of validation errors.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Validate data configuration
    if config.data.train_test_split <= 0 or config.data.train_test_split >= 1:
        errors.append("train_test_split must be between 0 and 1")
    
    if config.data.validation_split <= 0 or config.data.validation_split >= 1:
        errors.append("validation_split must be between 0 and 1")
    
    # Validate GAN configuration
    if config.gan.latent_dim <= 0:
        errors.append("latent_dim must be positive")
    
    if config.gan.batch_size <= 0:
        errors.append("batch_size must be positive")
    
    if config.gan.num_epochs <= 0:
        errors.append("num_epochs must be positive")
    
    # Validate learning rates
    if config.gan.learning_rate_generator <= 0:
        errors.append("learning_rate_generator must be positive")
    
    if config.gan.learning_rate_discriminator <= 0:
        errors.append("learning_rate_discriminator must be positive")
    
    # Validate macro scenarios
    baseline_vars = set(config.macro.baseline_scenario.keys())
    for scenario_name, scenario_values in config.macro.stress_scenarios.items():
        scenario_vars = set(scenario_values.keys())
        if scenario_vars != baseline_vars:
            errors.append(f"Scenario '{scenario_name}' has different variables than baseline")
    
    # Validate PD model configuration
    valid_model_types = ['logistic', 'gradient_boosting', 'macro_conditioned']
    for model_type in config.pd_model.model_types:
        if model_type not in valid_model_types:
            errors.append(f"Invalid model type: {model_type}")
    
    # Validate evaluation configuration
    if config.evaluation.lgd <= 0 or config.evaluation.lgd > 1:
        errors.append("LGD must be between 0 and 1")
    
    if config.evaluation.current_capital_ratio <= 0:
        errors.append("current_capital_ratio must be positive")
    
    return errors


# Example usage and default configuration file creation
if __name__ == "__main__":
    # Create default configuration
    default_config = get_default_config()
    
    # Save default configuration files
    default_config.save_yaml("config/default_config.yaml")
    default_config.save_json("config/default_config.json")
    
    # Validate configuration
    errors = validate_config(default_config)
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration validation passed!")
    
    print(f"Default configuration created with:")
    print(f"  - {len(default_config.data.numerical_features)} numerical features")
    print(f"  - {len(default_config.macro.stress_scenarios)} stress scenarios")
    print(f"  - {len(default_config.pd_model.model_types)} PD model types")

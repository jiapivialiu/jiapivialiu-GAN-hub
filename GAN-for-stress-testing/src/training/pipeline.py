"""
Training pipeline for GAN-based stress testing.

This module implements the complete training pipeline that orchestrates
data processing, GAN training, PD model training, and evaluation.
"""

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

from data.preprocessing import DataProcessor
from models.conditional_gan import ConditionalGAN
from models.pd_models import LogisticPDModel, GradientBoostingPDModel, MacroConditionedPDModel
from evaluation.metrics import generate_evaluation_report, plot_evaluation_results
from config.config import Config, setup_logging, validate_config

logger = logging.getLogger(__name__)


class StressTestingPipeline:
    """
    Complete pipeline for GAN-based stress testing.
    
    Manages the entire workflow from data loading to model evaluation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.setup_environment()
        
        # Initialize components
        self.data_processor = None
        self.gan_model = None
        self.pd_models = {}
        
        # Data storage
        self.data = {}
        self.synthetic_data = {}
        self.evaluation_results = {}
        
        # Training state
        self.training_history = {
            'gan_losses': [],
            'pd_model_scores': {},
            'evaluation_metrics': []
        }
    
    def setup_environment(self):
        """Setup logging and create necessary directories."""
        setup_logging(self.config)
        
        # Create directories
        for path in [self.config.model_save_path, self.config.results_save_path, 
                     self.config.logs_path, self.config.data.processed_data_path,
                     self.config.data.synthetic_data_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Set random seeds
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
        
        logger.info("Environment setup complete")
    
    def validate_configuration(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid
        """
        errors = validate_config(self.config)
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def load_and_process_data(self, data_path: str = None) -> Dict[str, Any]:
        """
        Load and process the raw data.
        
        Args:
            data_path: Path to raw data file
            
        Returns:
            Dictionary containing processed data
        """
        logger.info("Starting data loading and processing...")
        
        # Initialize data processor
        self.data_processor = DataProcessor()
        
        if data_path:
            # Load real data
            logger.info(f"Loading data from {data_path}")
            raw_data = pd.read_csv(data_path)
            
            # Process the data through the pipeline
            cleaned_data = self.data_processor.clean_data(raw_data)
            featured_data = self.data_processor.engineer_features(cleaned_data)
            processed_data = featured_data
        else:
            # Generate synthetic data for demonstration
            logger.info("Generating synthetic data for demonstration")
            processed_data = DataProcessor.create_synthetic_data(
                n_samples=10000,
                random_state=42
            )
        
        # Split data
        train_data, test_data = train_test_split(
            processed_data, 
            test_size=1-self.config.data.train_test_split,
            random_state=42
        )
        
        # Store processed data
        self.data = {
            'train': train_data,
            'test': test_data,
            'full': processed_data
        }
        
        logger.info(f"Data processing complete. Train: {len(train_data)}, Test: {len(test_data)}")
        return self.data
    
    def train_gan(self) -> Dict[str, Any]:
        """
        Train the Conditional GAN.
        
        Returns:
            Training history and metrics
        """
        logger.info("Starting GAN training...")
        
        if self.data_processor is None or 'train' not in self.data:
            raise ValueError("Data must be loaded and processed before training GAN")
        
        # Prepare training data
        train_features = self.data['train'][self.config.data.numerical_features].values
        train_features = torch.FloatTensor(train_features)
        
        # Create macro conditions for training (baseline scenario)
        baseline_condition = np.array(list(self.config.macro.baseline_scenario.values()))
        train_conditions = np.tile(baseline_condition, (len(train_features), 1))
        train_conditions = torch.FloatTensor(train_conditions)
        
        # Initialize GAN
        borrower_dim = len(self.config.data.numerical_features)
        macro_dim = len(self.config.macro.macro_variables)
        
        self.gan_model = ConditionalGAN(
            noise_dim=self.config.gan.latent_dim,
            borrower_dim=borrower_dim,
            macro_dim=macro_dim,
            generator_lr=self.config.gan.learning_rate_generator,
            discriminator_lr=self.config.gan.learning_rate_discriminator,
            device=self.config.device
        )
        
        # Train GAN
        logger.info("Starting GAN training loop...")
        training_history = {'generator_losses': [], 'discriminator_losses': []}
        
        dataset = torch.utils.data.TensorDataset(train_features, train_conditions)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.gan.batch_size, 
            shuffle=True
        )
        
        # Early stopping variables
        best_combined_loss = float('inf')
        patience_counter = 0
        patience = self.config.gan.patience
        min_delta = self.config.gan.min_delta
        
        for epoch in range(self.config.gan.num_epochs):
            epoch_g_losses = []
            epoch_d_losses = []
            
            for batch_idx, (real_data, conditions) in enumerate(dataloader):
                real_data = real_data.to(self.config.device)
                conditions = conditions.to(self.config.device)
                
                # Train step
                d_loss, g_loss = self.gan_model.train_step(real_data, conditions)
                
                epoch_d_losses.append(d_loss)
                epoch_g_losses.append(g_loss)
            
            # Record epoch averages
            avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
            avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
            
            training_history['discriminator_losses'].append(avg_d_loss)
            training_history['generator_losses'].append(avg_g_loss)
            
            # Combined loss for early stopping (balance between D and G)
            combined_loss = avg_g_loss + abs(avg_d_loss - 0.5)  # Target D_loss around 0.5
            
            # Early stopping logic
            if combined_loss < best_combined_loss - min_delta:
                best_combined_loss = combined_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check for problematic training patterns
            if avg_d_loss < 0.001:  # Discriminator collapse
                logger.warning(f"Epoch {epoch}: Discriminator collapse detected (D_loss={avg_d_loss:.4f})")
            if avg_g_loss > 20.0:  # Generator explosion
                logger.warning(f"Epoch {epoch}: Generator loss explosion detected (G_loss={avg_g_loss:.4f})")
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}, Combined={combined_loss:.4f}, Patience={patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best combined loss: {best_combined_loss:.4f}")
                break
            
            # Emergency stop for extreme cases
            if avg_d_loss < 0.0001 and avg_g_loss > 25.0:
                logger.warning(f"Emergency stop at epoch {epoch}: Training instability detected")
                break
        
        logger.info("GAN training completed")
        
        # Store training history
        self.training_history['gan_losses'] = training_history
        
        # Save trained model
        model_path = Path(self.config.model_save_path) / "conditional_gan.pt"
        self.gan_model.save_model(str(model_path))
        
        logger.info(f"GAN training complete. Model saved to {model_path}")
        return training_history
    
    def generate_synthetic_scenarios(self) -> Dict[str, np.ndarray]:
        """
        Generate synthetic data for all scenarios.
        
        Returns:
            Dictionary mapping scenario names to synthetic data
        """
        logger.info("Generating synthetic data for all scenarios...")
        
        if self.gan_model is None:
            raise ValueError("GAN model must be trained before generating synthetic data")
        
        synthetic_scenarios = {}
        n_samples = len(self.data['test'])  # Generate same number as test set
        
        # Generate for baseline scenario
        baseline_condition = np.array(list(self.config.macro.baseline_scenario.values()))
        baseline_data = self.gan_model.generate_samples(
            n_samples=n_samples,
            conditions=np.tile(baseline_condition, (n_samples, 1))
        )
        synthetic_scenarios['baseline'] = baseline_data
        
        # Generate for stress scenarios
        for scenario_name, scenario_values in self.config.macro.stress_scenarios.items():
            stress_condition = np.array(list(scenario_values.values()))
            stress_data = self.gan_model.generate_samples(
                n_samples=n_samples,
                conditions=np.tile(stress_condition, (n_samples, 1))
            )
            synthetic_scenarios[scenario_name] = stress_data
        
        # Store synthetic data
        self.synthetic_data = synthetic_scenarios
        
        # Save synthetic data
        for scenario, data in synthetic_scenarios.items():
            save_path = Path(self.config.data.synthetic_data_path) / f"synthetic_{scenario}.csv"
            df = pd.DataFrame(data, columns=self.config.data.numerical_features)
            df.to_csv(save_path, index=False)
            logger.info(f"Synthetic data for {scenario} saved to {save_path}")
        
        return synthetic_scenarios
    
    def train_pd_models(self) -> Dict[str, Any]:
        """
        Train PD models on real data.
        
        Returns:
            Dictionary containing trained models and their scores
        """
        logger.info("Training PD models...")
        
        # Prepare training data
        X_train = self.data['train'][self.config.data.numerical_features]
        y_train = self.data['train'][self.config.data.target_feature]
        X_test = self.data['test'][self.config.data.numerical_features]
        y_test = self.data['test'][self.config.data.target_feature]
        
        model_scores = {}
        
        # Train each model type
        for model_type in self.config.pd_model.model_types:
            logger.info(f"Training {model_type} model...")
            
            if model_type == 'logistic':
                model = LogisticPDModel(
                    max_iter=self.config.pd_model.logistic_max_iter,
                    C=self.config.pd_model.logistic_C,
                    solver=self.config.pd_model.logistic_solver
                )
            elif model_type == 'gradient_boosting':
                model = GradientBoostingPDModel(
                    n_estimators=self.config.pd_model.gb_n_estimators,
                    max_depth=self.config.pd_model.gb_max_depth,
                    learning_rate=self.config.pd_model.gb_learning_rate,
                    subsample=self.config.pd_model.gb_subsample,
                    random_state=self.config.pd_model.gb_random_state
                )
            elif model_type == 'macro_conditioned':
                # Add macro variables to features for macro-conditioned model
                baseline_macro = np.array(list(self.config.macro.baseline_scenario.values()))
                X_train_macro = np.column_stack([
                    X_train.values,
                    np.tile(baseline_macro, (len(X_train), 1))
                ])
                X_test_macro = np.column_stack([
                    X_test.values,
                    np.tile(baseline_macro, (len(X_test), 1))
                ])
                
                model = MacroConditionedPDModel()
                model.fit(X_train_macro, y_train)
                
                # Store model and evaluate
                self.pd_models[model_type] = model
                test_score = model.score(X_test_macro, y_test)
                model_scores[model_type] = {
                    'test_score': test_score,
                    'feature_importance': getattr(model.model, 'feature_importances_', None)
                }
                continue
            
            # Fit standard models
            model.fit(X_train, y_train)
            
            # Store model and evaluate
            self.pd_models[model_type] = model
            test_score = model.score(X_test, y_test)
            model_scores[model_type] = {
                'test_score': test_score,
                'feature_importance': getattr(model.model, 'feature_importances_', None)
            }
            
            logger.info(f"{model_type} model test score: {test_score:.4f}")
        
        # Store model scores
        self.training_history['pd_model_scores'] = model_scores
        
        # Save models
        models_path = Path(self.config.model_save_path) / "pd_models.pkl"
        with open(models_path, 'wb') as f:
            pickle.dump(self.pd_models, f)
        
        logger.info(f"PD models training complete. Models saved to {models_path}")
        return model_scores
    
    def evaluate_stress_testing(self) -> Dict[str, Any]:
        """
        Evaluate stress testing results.
        
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Evaluating stress testing results...")
        
        if not self.pd_models or not self.synthetic_data:
            raise ValueError("PD models and synthetic data required for evaluation")
        
        # Get the best performing PD model
        best_model_name = max(
            self.training_history['pd_model_scores'].keys(),
            key=lambda x: self.training_history['pd_model_scores'][x]['test_score']
        )
        best_model = self.pd_models[best_model_name]
        
        logger.info(f"Using {best_model_name} model for stress testing evaluation")
        
        # Calculate PDs for all scenarios
        scenario_pds = {}
        
        for scenario_name, synthetic_data in self.synthetic_data.items():
            if hasattr(best_model, 'predict_proba'):
                pds = best_model.predict_proba(synthetic_data)[:, 1]  # Probability of default
            else:
                pds = best_model.predict(synthetic_data)
            scenario_pds[scenario_name] = pds
        
        # Get real data for comparison
        real_test_data = self.data['test'][self.config.data.numerical_features].values
        
        # Generate comprehensive evaluation report
        evaluation_results = generate_evaluation_report(
            real_data=real_test_data,
            synthetic_data=self.synthetic_data['baseline'],
            baseline_pds=scenario_pds['baseline'],
            stressed_pds=scenario_pds.get('adverse', scenario_pds['baseline']),
            scenarios_pds=scenario_pds
        )
        
        # Store evaluation results
        self.evaluation_results = evaluation_results
        
        # Save evaluation results
        results_path = Path(self.config.results_save_path) / "evaluation_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(evaluation_results)
            json.dump(serializable_results, f, indent=2)
        
        # Generate plots if requested
        if self.config.evaluation.save_plots:
            plot_path = Path(self.config.results_save_path) / f"evaluation_plots.{self.config.evaluation.plot_format}"
            plot_evaluation_results(
                real_data=real_test_data,
                synthetic_data=self.synthetic_data['baseline'],
                feature_names=self.config.data.numerical_features,
                save_path=str(plot_path)
            )
        
        logger.info(f"Stress testing evaluation complete. Results saved to {results_path}")
        return evaluation_results
    
    def run_full_pipeline(self, data_path: str = None) -> Dict[str, Any]:
        """
        Run the complete stress testing pipeline.
        
        Args:
            data_path: Path to raw data file (optional)
            
        Returns:
            Dictionary containing all results
        """
        start_time = datetime.now()
        logger.info("Starting full stress testing pipeline...")
        
        # Validate configuration
        if not self.validate_configuration():
            raise ValueError("Configuration validation failed")
        
        try:
            # Step 1: Load and process data
            self.load_and_process_data(data_path)
            
            # Step 2: Train GAN
            self.train_gan()
            
            # Step 3: Generate synthetic scenarios
            self.generate_synthetic_scenarios()
            
            # Step 4: Train PD models
            self.train_pd_models()
            
            # Step 5: Evaluate stress testing
            self.evaluate_stress_testing()
            
            # Compile final results
            results = {
                'data_info': {
                    'train_samples': len(self.data['train']),
                    'test_samples': len(self.data['test']),
                    'features': self.config.data.numerical_features
                },
                'gan_training': self.training_history['gan_losses'],
                'pd_models': self.training_history['pd_model_scores'],
                'evaluation': self.evaluation_results,
                'synthetic_scenarios': list(self.synthetic_data.keys()),
                'runtime': str(datetime.now() - start_time)
            }
            
            # Save final results
            final_results_path = Path(self.config.results_save_path) / "final_results.json"
            with open(final_results_path, 'w') as f:
                serializable_results = self._make_json_serializable(results)
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Full pipeline complete! Total runtime: {results['runtime']}")
            logger.info(f"Final results saved to {final_results_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def load_trained_models(self, models_path: str = None):
        """Load previously trained models."""
        if models_path is None:
            models_path = Path(self.config.model_save_path) / "pd_models.pkl"
        
        with open(models_path, 'rb') as f:
            self.pd_models = pickle.load(f)
        
        # Load GAN model
        gan_path = Path(self.config.model_save_path) / "conditional_gan.pt"
        if gan_path.exists():
            # Initialize GAN with same architecture
            feature_dim = len(self.config.data.numerical_features)
            condition_dim = len(self.config.macro.macro_variables)
            
            self.gan_model = ConditionalGAN(
                feature_dim=feature_dim,
                condition_dim=condition_dim,
                latent_dim=self.config.gan.latent_dim,
                generator_layers=self.config.gan.generator_layers,
                discriminator_layers=self.config.gan.discriminator_layers,
                dropout_rate=self.config.gan.dropout_rate
            )
            self.gan_model.load_model(str(gan_path))
        
        logger.info("Trained models loaded successfully")
    
    def generate_stress_test_report(self) -> str:
        """
        Generate a formatted stress test report.
        
        Returns:
            Formatted report string
        """
        if not self.evaluation_results:
            raise ValueError("Evaluation must be run before generating report")
        
        report = []
        report.append("=" * 60)
        report.append("GAN-BASED STRESS TESTING REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data summary
        report.append("DATA SUMMARY:")
        report.append("-" * 20)
        if 'train' in self.data:
            report.append(f"Training samples: {len(self.data['train']):,}")
            report.append(f"Test samples: {len(self.data['test']):,}")
        report.append(f"Features: {len(self.config.data.numerical_features)}")
        report.append("")
        
        # Model performance
        report.append("MODEL PERFORMANCE:")
        report.append("-" * 20)
        for model_name, scores in self.training_history['pd_model_scores'].items():
            report.append(f"{model_name}: {scores['test_score']:.4f}")
        report.append("")
        
        # Stress testing results
        if 'stress_testing' in self.evaluation_results:
            stress_results = self.evaluation_results['stress_testing']
            
            report.append("STRESS TESTING RESULTS:")
            report.append("-" * 25)
            
            if 'pd_shift' in stress_results:
                pd_shift = stress_results['pd_shift']
                report.append(f"Baseline mean PD: {pd_shift['mean_pd_baseline']:.2%}")
                report.append(f"Stressed mean PD: {pd_shift['mean_pd_stressed']:.2%}")
                report.append(f"PD increase: {pd_shift['relative_pd_increase']:.1%}")
                report.append("")
            
            if 'ecl_impact' in stress_results:
                ecl_impact = stress_results['ecl_impact']
                report.append(f"ECL increase: {ecl_impact['relative_ecl_increase']:.1%}")
                report.append("")
            
            if 'capital_impact' in stress_results:
                capital_impact = stress_results['capital_impact']
                report.append(f"Capital ratio impact: {capital_impact['capital_buffer_impact']:.2%}")
        
        report.append("=" * 60)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    from config.config import get_default_config
    
    # Load configuration
    config = get_default_config()
    
    # Initialize pipeline
    pipeline = StressTestingPipeline(config)
    
    # Run full pipeline
    results = pipeline.run_full_pipeline()
    
    # Generate report
    report = pipeline.generate_stress_test_report()
    print(report)

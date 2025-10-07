"""
Evaluation metrics for GAN-based stress testing.

This module implements various metrics to evaluate both GAN quality
and stress testing effectiveness.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import logging

logger = logging.getLogger(__name__)


class GANQualityMetrics:
    """
    Metrics to evaluate the quality of generated synthetic data.
    
    Focuses on distributional similarity and structural preservation.
    """
    
    @staticmethod
    def kolmogorov_smirnov_test(real_data: np.ndarray, 
                               synthetic_data: np.ndarray) -> Dict[str, float]:
        """
        Perform Kolmogorov-Smirnov test for distributional similarity.
        
        Args:
            real_data: Real data samples [n_samples, n_features]
            synthetic_data: Synthetic data samples [n_samples, n_features]
            
        Returns:
            Dictionary with KS statistics and p-values for each feature
        """
        n_features = real_data.shape[1]
        results = {}
        
        for i in range(n_features):
            ks_stat, p_value = stats.ks_2samp(real_data[:, i], synthetic_data[:, i])
            results[f'feature_{i}'] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'similar': p_value > 0.05  # Accept null hypothesis at 5% level
            }
        
        # Overall similarity score
        avg_ks = np.mean([r['ks_statistic'] for r in results.values()])
        results['overall'] = {
            'avg_ks_statistic': avg_ks,
            'similarity_score': 1 - avg_ks  # Higher is better
        }
        
        return results
    
    @staticmethod
    def wasserstein_distances(real_data: np.ndarray, 
                             synthetic_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate Wasserstein distances between real and synthetic distributions.
        
        Args:
            real_data: Real data samples [n_samples, n_features]
            synthetic_data: Synthetic data samples [n_samples, n_features]
            
        Returns:
            Dictionary with Wasserstein distances for each feature
        """
        n_features = real_data.shape[1]
        distances = {}
        
        for i in range(n_features):
            distance = wasserstein_distance(real_data[:, i], synthetic_data[:, i])
            distances[f'feature_{i}'] = distance
        
        distances['mean_distance'] = np.mean(list(distances.values())[:-1])  # Exclude mean from mean calculation
        
        return distances
    
    @staticmethod
    def correlation_preservation(real_data: np.ndarray, 
                               synthetic_data: np.ndarray) -> Dict[str, float]:
        """
        Evaluate how well the synthetic data preserves correlation structure.
        
        Args:
            real_data: Real data samples [n_samples, n_features]
            synthetic_data: Synthetic data samples [n_samples, n_features]
            
        Returns:
            Dictionary with correlation preservation metrics
        """
        real_corr = np.corrcoef(real_data.T)
        synthetic_corr = np.corrcoef(synthetic_data.T)
        
        # Flatten correlation matrices (excluding diagonal)
        mask = ~np.eye(real_corr.shape[0], dtype=bool)
        real_corr_flat = real_corr[mask]
        synthetic_corr_flat = synthetic_corr[mask]
        
        # Calculate correlation between correlation matrices
        corr_correlation = np.corrcoef(real_corr_flat, synthetic_corr_flat)[0, 1]
        
        # Mean absolute difference
        mean_abs_diff = np.mean(np.abs(real_corr_flat - synthetic_corr_flat))
        
        return {
            'correlation_correlation': corr_correlation,
            'mean_abs_correlation_diff': mean_abs_diff,
            'correlation_preservation_score': corr_correlation
        }
    
    @staticmethod
    def mode_collapse_detection(synthetic_data: np.ndarray, 
                               threshold: float = 0.1) -> Dict[str, any]:
        """
        Detect mode collapse in generated data.
        
        Args:
            synthetic_data: Synthetic data samples [n_samples, n_features]
            threshold: Threshold for detecting mode collapse
            
        Returns:
            Dictionary with mode collapse metrics
        """
        n_samples, n_features = synthetic_data.shape
        
        # Calculate coefficient of variation for each feature
        cv_scores = []
        for i in range(n_features):
            mean_val = np.mean(synthetic_data[:, i])
            std_val = np.std(synthetic_data[:, i])
            cv = std_val / (abs(mean_val) + 1e-8)  # Add small epsilon to avoid division by zero
            cv_scores.append(cv)
        
        cv_scores = np.array(cv_scores)
        
        # Features with low CV might indicate mode collapse
        low_variance_features = np.sum(cv_scores < threshold)
        
        return {
            'cv_scores': cv_scores,
            'mean_cv': np.mean(cv_scores),
            'low_variance_features': low_variance_features,
            'mode_collapse_ratio': low_variance_features / n_features,
            'likely_mode_collapse': low_variance_features > n_features * 0.3
        }


class StressTestingMetrics:
    """
    Metrics to evaluate stress testing effectiveness and risk assessment.
    """
    
    @staticmethod
    def portfolio_pd_shift(baseline_pds: np.ndarray, 
                          stressed_pds: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio PD shift metrics.
        
        Args:
            baseline_pds: Baseline probability of default [n_borrowers]
            stressed_pds: Stressed probability of default [n_borrowers]
            
        Returns:
            Dictionary with PD shift metrics
        """
        return {
            'mean_pd_baseline': np.mean(baseline_pds),
            'mean_pd_stressed': np.mean(stressed_pds),
            'mean_pd_shift': np.mean(stressed_pds) - np.mean(baseline_pds),
            'relative_pd_increase': (np.mean(stressed_pds) / np.mean(baseline_pds)) - 1,
            'median_pd_baseline': np.median(baseline_pds),
            'median_pd_stressed': np.median(stressed_pds),
            'p95_pd_baseline': np.percentile(baseline_pds, 95),
            'p95_pd_stressed': np.percentile(stressed_pds, 95),
            'p99_pd_baseline': np.percentile(baseline_pds, 99),
            'p99_pd_stressed': np.percentile(stressed_pds, 99)
        }
    
    @staticmethod
    def expected_credit_loss_impact(baseline_pds: np.ndarray,
                                  stressed_pds: np.ndarray,
                                  lgd: float = 0.45,
                                  ead: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate Expected Credit Loss (ECL) impact.
        
        Args:
            baseline_pds: Baseline PDs [n_borrowers]
            stressed_pds: Stressed PDs [n_borrowers]
            lgd: Loss Given Default (assumed constant)
            ead: Exposure at Default [n_borrowers] (if None, assumes 1.0)
            
        Returns:
            Dictionary with ECL impact metrics
        """
        if ead is None:
            ead = np.ones_like(baseline_pds)
        
        # Calculate ECL = PD * LGD * EAD
        ecl_baseline = baseline_pds * lgd * ead
        ecl_stressed = stressed_pds * lgd * ead
        
        total_ecl_baseline = np.sum(ecl_baseline)
        total_ecl_stressed = np.sum(ecl_stressed)
        
        return {
            'total_ecl_baseline': total_ecl_baseline,
            'total_ecl_stressed': total_ecl_stressed,
            'ecl_increase': total_ecl_stressed - total_ecl_baseline,
            'relative_ecl_increase': (total_ecl_stressed / total_ecl_baseline) - 1,
            'mean_ecl_baseline': np.mean(ecl_baseline),
            'mean_ecl_stressed': np.mean(ecl_stressed),
            'total_exposure': np.sum(ead)
        }
    
    @staticmethod
    def scenario_discrimination(scenarios_pds: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate how well the model discriminates between different scenarios.
        
        Args:
            scenarios_pds: Dictionary mapping scenario names to PD arrays
            
        Returns:
            Dictionary with discrimination metrics
        """
        scenario_names = list(scenarios_pds.keys())
        mean_pds = {name: np.mean(pds) for name, pds in scenarios_pds.items()}
        
        # Calculate coefficient of variation across scenarios
        scenario_means = list(mean_pds.values())
        cv_scenarios = np.std(scenario_means) / np.mean(scenario_means)
        
        # Pairwise differences
        pairwise_diffs = {}
        for i, name1 in enumerate(scenario_names):
            for j, name2 in enumerate(scenario_names[i+1:], i+1):
                diff = mean_pds[name2] - mean_pds[name1]
                pairwise_diffs[f'{name1}_vs_{name2}'] = diff
        
        return {
            'scenario_means': mean_pds,
            'coefficient_variation': cv_scenarios,
            'max_scenario_diff': max(pairwise_diffs.values()) if pairwise_diffs else 0,
            'min_scenario_diff': min(pairwise_diffs.values()) if pairwise_diffs else 0,
            'pairwise_differences': pairwise_diffs,
            'good_discrimination': cv_scenarios > 0.1  # Arbitrary threshold
        }
    
    @staticmethod
    def capital_adequacy_impact(baseline_pds: np.ndarray,
                              stressed_pds: np.ndarray,
                              risk_weights: np.ndarray = None,
                              current_capital_ratio: float = 0.12) -> Dict[str, float]:
        """
        Estimate capital adequacy impact from stress testing.
        
        Args:
            baseline_pds: Baseline PDs [n_borrowers]
            stressed_pds: Stressed PDs [n_borrowers]
            risk_weights: Risk weights for each exposure [n_borrowers]
            current_capital_ratio: Current capital adequacy ratio
            
        Returns:
            Dictionary with capital impact metrics
        """
        if risk_weights is None:
            # Simple risk weight estimation based on PD
            risk_weights = np.where(baseline_pds < 0.01, 0.5, 
                                  np.where(baseline_pds < 0.05, 1.0, 1.5))
        
        # Calculate risk-weighted assets (simplified)
        rwa_baseline = np.sum(risk_weights * baseline_pds)
        rwa_stressed = np.sum(risk_weights * stressed_pds)
        
        # Estimate capital impact (simplified)
        capital_impact = rwa_stressed - rwa_baseline
        new_capital_ratio = current_capital_ratio - (capital_impact / rwa_baseline) * current_capital_ratio
        
        return {
            'rwa_baseline': rwa_baseline,
            'rwa_stressed': rwa_stressed,
            'rwa_increase': rwa_stressed - rwa_baseline,
            'relative_rwa_increase': (rwa_stressed / rwa_baseline) - 1,
            'current_capital_ratio': current_capital_ratio,
            'estimated_new_capital_ratio': new_capital_ratio,
            'capital_buffer_impact': current_capital_ratio - new_capital_ratio
        }


def generate_evaluation_report(real_data: np.ndarray,
                             synthetic_data: np.ndarray,
                             baseline_pds: np.ndarray,
                             stressed_pds: np.ndarray,
                             scenarios_pds: Dict[str, np.ndarray] = None) -> Dict[str, any]:
    """
    Generate comprehensive evaluation report.
    
    Args:
        real_data: Real borrower data
        synthetic_data: Synthetic borrower data
        baseline_pds: Baseline PD estimates
        stressed_pds: Stressed PD estimates
        scenarios_pds: Multiple scenario PD estimates
        
    Returns:
        Comprehensive evaluation report
    """
    report = {}
    
    # GAN Quality Metrics
    logger.info("Computing GAN quality metrics...")
    report['gan_quality'] = {
        'ks_test': GANQualityMetrics.kolmogorov_smirnov_test(real_data, synthetic_data),
        'wasserstein_distances': GANQualityMetrics.wasserstein_distances(real_data, synthetic_data),
        'correlation_preservation': GANQualityMetrics.correlation_preservation(real_data, synthetic_data),
        'mode_collapse': GANQualityMetrics.mode_collapse_detection(synthetic_data)
    }
    
    # Stress Testing Metrics
    logger.info("Computing stress testing metrics...")
    report['stress_testing'] = {
        'pd_shift': StressTestingMetrics.portfolio_pd_shift(baseline_pds, stressed_pds),
        'ecl_impact': StressTestingMetrics.expected_credit_loss_impact(baseline_pds, stressed_pds),
        'capital_impact': StressTestingMetrics.capital_adequacy_impact(baseline_pds, stressed_pds)
    }
    
    if scenarios_pds:
        report['stress_testing']['scenario_discrimination'] = \
            StressTestingMetrics.scenario_discrimination(scenarios_pds)
    
    return report


def plot_evaluation_results(real_data: np.ndarray,
                          synthetic_data: np.ndarray,
                          feature_names: List[str] = None,
                          save_path: str = None):
    """
    Create visualization plots for evaluation results.
    
    Args:
        real_data: Real data samples
        synthetic_data: Synthetic data samples
        feature_names: Names of features
        save_path: Path to save plots
    """
    n_features = real_data.shape[1]
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GAN Evaluation: Real vs Synthetic Data', fontsize=16)
    
    # Plot distributions for first 6 features
    for i in range(min(6, n_features)):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        ax.hist(real_data[:, i], bins=30, alpha=0.7, label='Real', density=True)
        ax.hist(synthetic_data[:, i], bins=30, alpha=0.7, label='Synthetic', density=True)
        ax.set_title(f'{feature_names[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Evaluation plots saved to {save_path}")
    
    plt.show()

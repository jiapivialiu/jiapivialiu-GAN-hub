#!/usr/bin/env python3
"""
Quick test of the GAN stress testing modules.
"""

import sys
import os
sys.path.append('src')

def test_modules():
    """Test importing our custom modules."""
    print("Testing GAN Stress Testing Modules")
    print("=" * 40)
    
    try:
        from config.config import get_default_config, Config
        print("Config module imported successfully")
        
        # Test configuration
        config = get_default_config()
        print(f"   - Project: {config.project_name}")
        print(f"   - Features: {len(config.data.numerical_features)}")
        print(f"   - Scenarios: {list(config.macro.stress_scenarios.keys())}")
        
    except Exception as e:
        print(f"Config module failed: {e}")
    
    try:
        from data.preprocessing import DataProcessor
        print("Data processing module imported successfully")
        
        # Test data processor
        processor = DataProcessor(get_default_config().data)
        print(f"   - Numerical features: {len(processor.config.numerical_features)}")
        
    except Exception as e:
        print(f"Data processing module failed: {e}")
    
    try:
        from models.conditional_gan import ConditionalGAN
        print("Conditional GAN module imported successfully")
        
    except Exception as e:
        print(f"Conditional GAN module failed: {e}")
    
    try:
        from models.pd_models import LogisticPDModel, GradientBoostingPDModel
        print("PD models module imported successfully")
        
    except Exception as e:
        print(f"PD models module failed: {e}")
    
    try:
        from evaluation.metrics import GANQualityMetrics, StressTestingMetrics
        print("Evaluation metrics module imported successfully")
        
    except Exception as e:
        print(f"Evaluation metrics module failed: {e}")
    
    try:
        from training.pipeline import StressTestingPipeline
        print("Training pipeline module imported successfully")
        
        # Test pipeline initialization
        pipeline = StressTestingPipeline(get_default_config())
        print(f"   - Pipeline initialized for project: {pipeline.config.project_name}")
        
    except Exception as e:
        print(f"Training pipeline module failed: {e}")
    
    print()
    print("Module testing complete!")
    print("Ready to run the demo notebook!")

if __name__ == "__main__":
    test_modules()

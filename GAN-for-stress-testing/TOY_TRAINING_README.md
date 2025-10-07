# Toy Training Debug Script

This directory contains a standalone toy training script for quick debugging and testing of GAN training optimizations.

## Files

- **`toy_training_debug.py`** - Main standalone training script
- **`test_toy_training.py`** - Simple test runner
- **`TOY_TRAINING_README.md`** - This file

## Quick Start

### 1. Basic Usage

Run from the `GAN-for-stress-testing` directory:

```bash
# Basic training (15 epochs, 1000 samples)
python toy_training_debug.py

# Quick test (5 epochs, 500 samples)
python toy_training_debug.py --epochs 5 --samples 500 --verbose

# Full debug with plots
python toy_training_debug.py --epochs 20 --verbose --plot
```

### 2. Test Runner

Use the test script for automated testing:

```bash
python test_toy_training.py
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 15 | Number of training epochs |
| `--samples` | 1000 | Number of toy data samples |
| `--batch-size` | 64 | Training batch size |
| `--verbose` | False | Show detailed training progress |
| `--plot` | False | Display training plots |
| `--seed` | 42 | Random seed for reproducibility |

## Example Outputs

### Successful Training
```
🧸 TOY GAN TRAINING DEBUG SCRIPT
==================================================
📊 Configuration:
   Epochs: 15
   Samples: 1000
   Batch size: 64

✅ Generated 1000 toy financial records
📊 Features: ['age', 'income_annual', 'dti_ratio', 'fico_score', 'loan_amount', 'employment_length', 'ltv_ratio']

🚀 FAST-CONVERGENCE TOY TRAINING
==================================================
🏋️ Training Configuration:
   Generator LR: 0.0002
   Discriminator LR: 3e-05
   Batch Size: 64
   Max Epochs: 15

📚 Phase 1: Generator Pre-training (3 epochs)
   Pre-train Epoch 0: G_loss=0.6821
   Pre-train Epoch 1: G_loss=0.6798
   Pre-train Epoch 2: G_loss=0.6776

⚔️ Phase 2: Progressive Adversarial Training
   Epoch  0: D_loss=0.6941, G_loss=0.6654, Stability=0.0287, Noise=0.1000
   Epoch  2: D_loss=0.6923, G_loss=0.6598, Stability=0.0325, Noise=0.0903
   🎯 Fast convergence at epoch 8!

✅ Training completed in 9 epochs
📊 Final G_loss: 0.6534
📊 Final D_loss: 0.6891

🎯 QUALITY EVALUATION
==============================
age            : Mean error   2.1%, Std error  14.2%
income_annual  : Mean error   1.8%, Std error  12.6%
dti_ratio      : Mean error   3.4%, Std error  15.1%
fico_score     : Mean error   0.9%, Std error  13.8%
loan_amount    : Mean error   2.7%, Std error  16.2%
employment_length: Mean error   4.1%, Std error  18.3%
ltv_ratio      : Mean error   1.2%, Std error  14.8%

📊 Overall Quality Score: 0.849
🏆 EXCELLENT quality!

🎉 TOY TRAINING COMPLETE!
📊 Epochs completed: 9
🎯 Final quality score: 0.849
📈 Final generator loss: 0.6534
✅ Training successful! Ready for production testing.
```

## Key Features

### 🚀 Fast Convergence Optimizations
1. **Generator Pre-training** - 3 epochs head start
2. **Adaptive Learning Rates** - Dynamic LR scheduling
3. **Curriculum Learning** - Progressive difficulty
4. **Balanced Training** - Generator trains 2x per discriminator
5. **Enhanced Loss Functions** - Feature matching + diversity loss
6. **Fast Convergence Detection** - Early stopping on quality metrics

### 🎯 Debugging Features
- **Synthetic Data Generation** - Creates realistic financial data
- **Quality Evaluation** - Statistical comparison metrics
- **Training Visualization** - Loss curves and stability plots
- **Configurable Parameters** - Easy hyperparameter testing
- **Verbose Logging** - Detailed training progress

### 📊 Generated Features
- **age** - Customer age (18-80)
- **income_annual** - Annual income ($20K-$150K)
- **dti_ratio** - Debt-to-income ratio (0.05-0.6)
- **fico_score** - Credit score (350-850)
- **loan_amount** - Loan amount ($50K-$500K)
- **employment_length** - Years employed (0-25)
- **ltv_ratio** - Loan-to-value ratio (0.3-1.0)

## Troubleshooting

### Import Errors
Make sure you're running from the `GAN-for-stress-testing` directory:
```bash
cd GAN-for-stress-testing
python toy_training_debug.py
```

### Poor Quality Results
Try adjusting hyperparameters:
```bash
# Longer training
python toy_training_debug.py --epochs 25

# Smaller batch size
python toy_training_debug.py --batch-size 32

# More samples
python toy_training_debug.py --samples 2000
```

### Memory Issues
Reduce parameters:
```bash
# Smaller dataset
python toy_training_debug.py --samples 500 --batch-size 32
```

## Integration with Main Project

This toy script extracts the optimized training logic from the main notebook. Key improvements from the toy training can be integrated back into:

- `src/training/pipeline.py` - Main training pipeline
- `src/models/conditional_gan.py` - GAN model improvements
- `notebooks/demo_stress_testing.ipynb` - Notebook updates

## Performance Benchmarks

Typical performance on CPU:
- **5 epochs, 500 samples**: ~30 seconds
- **15 epochs, 1000 samples**: ~2 minutes
- **25 epochs, 2000 samples**: ~8 minutes

Quality benchmarks:
- **Excellent (>0.8)**: Production ready
- **Good (0.6-0.8)**: Acceptable for testing
- **Poor (<0.6)**: Needs hyperparameter tuning

## Next Steps

1. **Test different hyperparameters** using the toy script
2. **Validate improvements** with the test runner
3. **Scale successful configs** to the main pipeline
4. **Integrate optimizations** into production code

This toy training script is designed for rapid iteration and debugging of GAN training improvements! 🚀

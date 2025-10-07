#!/usr/bin/env python3
"""
Toy GAN Training Script for Debugging

A standalone script to quickly test and debug GAN training with minimal setup.
This extracts the optimized fast-convergence training from the main notebook
for easy iteration and debugging.

Usage:
    python toy_training_debug.py [--epochs EPOCHS] [--verbose] [--plot]
    
Example:
    python toy_training_debug.py --epochs 15 --verbose --plot
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from models.conditional_gan import ConditionalGAN
    from config.config import Config
    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the GAN-for-stress-testing directory")
    sys.exit(1)


def generate_toy_data(n_samples=1000, seed=42):
    """Generate synthetic financial data for testing."""
    np.random.seed(seed)
    
    # Generate correlated financial features
    data = {}
    
    # Age: 18-80
    data['age'] = np.random.normal(40, 12, n_samples)
    data['age'] = np.clip(data['age'], 18, 80)
    
    # Income: correlated with age
    data['income_annual'] = 30000 + data['age'] * 1000 + np.random.normal(0, 15000, n_samples)
    data['income_annual'] = np.clip(data['income_annual'], 20000, 150000)
    
    # DTI ratio: inversely correlated with income
    data['dti_ratio'] = 0.4 - (data['income_annual'] - 50000) / 200000 + np.random.normal(0, 0.1, n_samples)
    data['dti_ratio'] = np.clip(data['dti_ratio'], 0.05, 0.6)
    
    # FICO score: correlated with income, inversely with DTI
    data['fico_score'] = 600 + (data['income_annual'] - 30000) / 1000 + (0.4 - data['dti_ratio']) * 300 + np.random.normal(0, 30, n_samples)
    data['fico_score'] = np.clip(data['fico_score'], 350, 850)
    
    # Loan amount: correlated with income
    data['loan_amount'] = data['income_annual'] * (2 + np.random.normal(0, 0.5, n_samples))
    data['loan_amount'] = np.clip(data['loan_amount'], 50000, 500000)
    
    # Employment length: correlated with age
    data['employment_length'] = (data['age'] - 18) / 3 + np.random.normal(0, 2, n_samples)
    data['employment_length'] = np.clip(data['employment_length'], 0, 25)
    
    # LTV ratio: somewhat random
    data['ltv_ratio'] = np.random.normal(0.8, 0.15, n_samples)
    data['ltv_ratio'] = np.clip(data['ltv_ratio'], 0.3, 1.0)
    
    df = pd.DataFrame(data)
    
    print(f"✅ Generated {n_samples} toy financial records")
    print(f"📊 Features: {list(df.columns)}")
    print(f"📈 Data shape: {df.shape}")
    
    return df


class ToyGANTrainer:
    """Standalone GAN trainer for debugging."""
    
    def __init__(self, data_features, config_params=None):
        """Initialize the toy trainer."""
        self.data_features = data_features
        self.feature_names = ['age', 'income_annual', 'dti_ratio', 'fico_score', 
                            'loan_amount', 'employment_length', 'ltv_ratio']
        
        # Default config
        self.config_params = config_params or {
            'generator_lr': 0.0002,
            'discriminator_lr': 0.00003,
            'batch_size': 64,
            'device': 'cpu'
        }
        
        # Data normalization
        self.scaler = StandardScaler()
        self.normalized_data = self.scaler.fit_transform(data_features)
        
        print(f"📊 Normalized data - Mean: {np.mean(self.normalized_data, axis=0)}")
        print(f"📊 Normalized data - Std: {np.std(self.normalized_data, axis=0)}")
        
        # Initialize GAN
        self.gan = ConditionalGAN(
            noise_dim=100,
            borrower_dim=len(self.feature_names),
            macro_dim=6,
            generator_lr=self.config_params['generator_lr'],
            discriminator_lr=self.config_params['discriminator_lr'],
            device=self.config_params['device']
        )
        
        print("✅ Toy GAN trainer initialized")
    
    def train_fast_convergence(self, max_epochs=15, verbose=True):
        """
        Train GAN with fast convergence optimizations.
        
        Returns training metrics for analysis.
        """
        print(f"\n🚀 FAST-CONVERGENCE TOY TRAINING")
        print("="*50)
        
        # Prepare data
        train_features = torch.FloatTensor(self.normalized_data)
        macro_conditions = torch.FloatTensor(np.random.randn(len(self.normalized_data), 6))
        
        dataset = torch.utils.data.TensorDataset(train_features, macro_conditions)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config_params['batch_size'], 
            shuffle=True
        )
        
        # Create optimizers
        g_optimizer = torch.optim.Adam(
            self.gan.generator.parameters(), 
            lr=self.config_params['generator_lr'], 
            betas=(0.5, 0.999)
        )
        d_optimizer = torch.optim.Adam(
            self.gan.discriminator.parameters(), 
            lr=self.config_params['discriminator_lr'], 
            betas=(0.5, 0.999)
        )
        
        # Learning rate schedulers
        g_scheduler = lr_scheduler.ReduceLROnPlateau(g_optimizer, mode='min', factor=0.8, patience=3)
        d_scheduler = lr_scheduler.ReduceLROnPlateau(d_optimizer, mode='min', factor=0.9, patience=5)
        
        # Training metrics
        metrics = {
            'generator_losses': [],
            'discriminator_losses': [],
            'stability_scores': [],
            'learning_rates': {'generator': [], 'discriminator': []}
        }
        
        print(f"🏋️ Training Configuration:")
        print(f"   Generator LR: {self.config_params['generator_lr']}")
        print(f"   Discriminator LR: {self.config_params['discriminator_lr']}")
        print(f"   Batch Size: {self.config_params['batch_size']}")
        print(f"   Max Epochs: {max_epochs}")
        
        # Phase 1: Generator Pre-training (3 epochs for toy model)
        if verbose:
            print(f"\n📚 Phase 1: Generator Pre-training (3 epochs)")
        
        for pre_epoch in range(3):
            epoch_g_losses = []
            
            for batch_real, batch_conditions in dataloader:
                batch_real = batch_real.to(self.config_params['device'])
                batch_conditions = batch_conditions.to(self.config_params['device'])
                
                # Generator training only
                noise = torch.randn(batch_real.size(0), 100, device=self.config_params['device'])
                fake_data = self.gan.generator(noise, batch_conditions)
                fake_pred = self.gan.discriminator(fake_data, batch_conditions)
                
                g_loss = self.gan.criterion(fake_pred, torch.ones_like(fake_pred))
                
                # Feature matching loss
                with torch.no_grad():
                    real_features = self.gan.discriminator(batch_real, batch_conditions)
                fake_features = self.gan.discriminator(fake_data, batch_conditions)
                feature_loss = nn.functional.mse_loss(fake_features, real_features)
                
                total_g_loss = g_loss + 0.1 * feature_loss
                
                g_optimizer.zero_grad()
                total_g_loss.backward()
                g_optimizer.step()
                
                epoch_g_losses.append(total_g_loss.item())
            
            if verbose:
                print(f"   Pre-train Epoch {pre_epoch}: G_loss={np.mean(epoch_g_losses):.4f}")
        
        # Phase 2: Progressive Adversarial Training
        if verbose:
            print(f"\n⚔️ Phase 2: Progressive Adversarial Training")
        
        best_g_loss = float('inf')
        patience = 5  # Faster convergence for toy model
        patience_counter = 0
        noise_scale = 0.1  # Curriculum learning
        
        for epoch in range(max_epochs):
            epoch_d_losses = []
            epoch_g_losses = []
            
            # Adaptive noise reduction
            current_noise_scale = noise_scale * (0.95 ** epoch)
            
            for batch_real, batch_conditions in dataloader:
                batch_real = batch_real.to(self.config_params['device'])
                batch_conditions = batch_conditions.to(self.config_params['device'])
                
                # Add curriculum learning noise
                if current_noise_scale > 0.001:
                    batch_real += torch.randn_like(batch_real) * current_noise_scale
                
                # Discriminator training (1 step)
                d_optimizer.zero_grad()
                
                real_pred = self.gan.discriminator(batch_real, batch_conditions)
                d_loss_real = self.gan.criterion(real_pred, torch.ones_like(real_pred))
                
                noise = torch.randn(batch_real.size(0), 100, device=self.config_params['device'])
                fake_data = self.gan.generator(noise, batch_conditions).detach()
                fake_pred = self.gan.discriminator(fake_data, batch_conditions)
                d_loss_fake = self.gan.criterion(fake_pred, torch.zeros_like(fake_pred))
                
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                d_optimizer.step()
                
                # Generator training (2 steps for faster convergence)
                for g_step in range(2):
                    g_optimizer.zero_grad()
                    
                    noise = torch.randn(batch_real.size(0), 100, device=self.config_params['device'])
                    fake_data = self.gan.generator(noise, batch_conditions)
                    fake_pred = self.gan.discriminator(fake_data, batch_conditions)
                    
                    g_loss = self.gan.criterion(fake_pred, torch.ones_like(fake_pred))
                    
                    # Feature matching
                    with torch.no_grad():
                        real_features = self.gan.discriminator(batch_real, batch_conditions)
                    fake_features = self.gan.discriminator(fake_data, batch_conditions)
                    feature_loss = nn.functional.mse_loss(fake_features, real_features)
                    
                    # Diversity loss
                    batch_size = fake_data.size(0)
                    if batch_size > 1:
                        fake_flat = fake_data.view(batch_size, -1)
                        diversity_loss = -torch.mean(torch.var(fake_flat, dim=0))
                    else:
                        diversity_loss = 0
                    
                    total_g_loss = g_loss + 0.1 * feature_loss + 0.01 * diversity_loss
                    total_g_loss.backward()
                    g_optimizer.step()
                
                epoch_d_losses.append(d_loss.item())
                epoch_g_losses.append(total_g_loss.item())
            
            # Calculate metrics
            avg_d_loss = np.mean(epoch_d_losses)
            avg_g_loss = np.mean(epoch_g_losses)
            stability_score = abs(avg_d_loss - avg_g_loss)
            
            # Store metrics
            metrics['generator_losses'].append(avg_g_loss)
            metrics['discriminator_losses'].append(avg_d_loss)
            metrics['stability_scores'].append(stability_score)
            metrics['learning_rates']['generator'].append(g_optimizer.param_groups[0]['lr'])
            metrics['learning_rates']['discriminator'].append(d_optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            g_scheduler.step(avg_g_loss)
            d_scheduler.step(avg_d_loss)
            
            # Early stopping
            if avg_g_loss < best_g_loss - 0.001:
                best_g_loss = avg_g_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch % 2 == 0 or epoch < 5):
                print(f"   Epoch {epoch:2d}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}, "
                      f"Stability={stability_score:.4f}, Noise={current_noise_scale:.4f}")
            
            # Fast convergence check
            if patience_counter >= patience:
                if verbose:
                    print(f"   🎯 Fast convergence at epoch {epoch}!")
                break
                
            if avg_g_loss < 0.5 and stability_score < 0.3:
                if verbose:
                    print(f"   🏆 High quality at epoch {epoch}!")
                break
        
        print(f"\n✅ Training completed in {len(metrics['generator_losses'])} epochs")
        print(f"📊 Final G_loss: {metrics['generator_losses'][-1]:.4f}")
        print(f"📊 Final D_loss: {metrics['discriminator_losses'][-1]:.4f}")
        
        return metrics
    
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic data using trained GAN."""
        noise = torch.randn(n_samples, 100)
        macro_conditions = torch.randn(n_samples, 6)
        
        with torch.no_grad():
            synthetic_normalized = self.gan.generator(noise, macro_conditions).cpu().numpy()
        
        # Denormalize
        synthetic_data = self.scaler.inverse_transform(synthetic_normalized)
        
        return synthetic_data
    
    def evaluate_quality(self, n_samples=1000, verbose=True):
        """Quick quality evaluation of generated data."""
        synthetic_data = self.generate_synthetic_data(n_samples)
        
        if verbose:
            print(f"\n🎯 QUALITY EVALUATION")
            print("="*30)
        
        total_mean_error = 0
        total_std_error = 0
        
        for i, feature_name in enumerate(self.feature_names):
            orig_mean = np.mean(self.data_features[:, i])
            synth_mean = np.mean(synthetic_data[:, i])
            mean_error = abs(orig_mean - synth_mean) / orig_mean
            
            orig_std = np.std(self.data_features[:, i])
            synth_std = np.std(synthetic_data[:, i])
            std_error = abs(orig_std - synth_std) / orig_std
            
            total_mean_error += mean_error
            total_std_error += std_error
            
            if verbose:
                print(f"{feature_name:15}: Mean error {mean_error*100:5.1f}%, Std error {std_error*100:5.1f}%")
        
        avg_mean_error = total_mean_error / len(self.feature_names)
        avg_std_error = total_std_error / len(self.feature_names)
        quality_score = max(0, 1 - (avg_mean_error + avg_std_error) / 2)
        
        if verbose:
            print(f"\n📊 Overall Quality Score: {quality_score:.3f}")
            
            if quality_score > 0.8:
                print("🏆 EXCELLENT quality!")
            elif quality_score > 0.6:
                print("👍 GOOD quality")
            else:
                print("⚠️ Needs improvement")
        
        return quality_score, synthetic_data
    
    def plot_training_progress(self, metrics):
        """Plot training progress."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('🚀 Toy GAN Training Progress', fontsize=14, fontweight='bold')
            
            epochs = range(len(metrics['generator_losses']))
            
            # Loss evolution
            axes[0, 0].plot(epochs, metrics['generator_losses'], 'r-', label='Generator', linewidth=2)
            axes[0, 0].plot(epochs, metrics['discriminator_losses'], 'b-', label='Discriminator', linewidth=2)
            axes[0, 0].set_title('Loss Evolution')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Stability score
            axes[0, 1].plot(epochs, metrics['stability_scores'], 'g-', linewidth=2)
            axes[0, 1].set_title('Training Stability')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Stability Score')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Learning rates
            axes[1, 0].plot(epochs, metrics['learning_rates']['generator'], 'r--', label='Generator LR')
            axes[1, 0].plot(epochs, metrics['learning_rates']['discriminator'], 'b--', label='Discriminator LR')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
            
            # Loss ratio
            loss_ratios = [g/d if d > 0 else 1 for g, d in zip(metrics['generator_losses'], metrics['discriminator_losses'])]
            axes[1, 1].plot(epochs, loss_ratios, 'purple', linewidth=2)
            axes[1, 1].axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Ideal Ratio')
            axes[1, 1].set_title('Generator/Discriminator Loss Ratio')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('G_loss / D_loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Plotting error: {e}")


def main():
    """Main function for toy training."""
    parser = argparse.ArgumentParser(description='Toy GAN Training for Debugging')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--samples', type=int, default=1000, help='Number of toy samples to generate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--plot', action='store_true', help='Show training plots')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("🧸 TOY GAN TRAINING DEBUG SCRIPT")
    print("="*50)
    print(f"📊 Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Samples: {args.samples}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Verbose: {args.verbose}")
    print(f"   Plotting: {args.plot}")
    print(f"   Seed: {args.seed}")
    
    # Generate toy data
    toy_df = generate_toy_data(n_samples=args.samples, seed=args.seed)
    toy_features = toy_df.values
    
    # Configuration
    config_params = {
        'generator_lr': 0.0002,
        'discriminator_lr': 0.00003,
        'batch_size': args.batch_size,
        'device': 'cpu'
    }
    
    # Initialize trainer
    trainer = ToyGANTrainer(toy_features, config_params)
    
    # Train GAN
    print(f"\n🚀 Starting toy training...")
    metrics = trainer.train_fast_convergence(max_epochs=args.epochs, verbose=args.verbose)
    
    # Evaluate quality
    quality_score, synthetic_data = trainer.evaluate_quality(verbose=args.verbose)
    
    # Show plots if requested
    if args.plot:
        trainer.plot_training_progress(metrics)
    
    # Summary
    print(f"\n🎉 TOY TRAINING COMPLETE!")
    print(f"📊 Epochs completed: {len(metrics['generator_losses'])}")
    print(f"🎯 Final quality score: {quality_score:.3f}")
    print(f"📈 Final generator loss: {metrics['generator_losses'][-1]:.4f}")
    
    if quality_score > 0.7:
        print("✅ Training successful! Ready for production testing.")
    else:
        print("⚠️ Training needs improvement. Check hyperparameters.")
    
    return metrics, quality_score


if __name__ == "__main__":
    main()

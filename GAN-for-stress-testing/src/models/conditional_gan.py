"""
Conditional GAN for Credit Risk Stress Testing

This module implements a conditional GAN that generates synthetic borrower profiles
conditioned on macroeconomic stress scenarios for credit risk modeling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """
    Generator network for conditional GAN.
    
    Generates synthetic borrower features conditioned on macroeconomic indicators.
    
    Args:
        noise_dim: Dimension of input noise vector
        borrower_dim: Dimension of borrower features to generate
        macro_dim: Dimension of macroeconomic conditioning variables
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self, 
        noise_dim: int = 100,
        borrower_dim: int = 15,
        macro_dim: int = 6,
        hidden_dims: list = [256, 512, 256]
    ):
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.borrower_dim = borrower_dim
        self.macro_dim = macro_dim
        
        # Input dimension is noise + macro conditions
        input_dim = noise_dim + macro_dim
        
        # Build generator layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, borrower_dim))
        layers.append(nn.Tanh())  # Normalize outputs to [-1, 1]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, noise: torch.Tensor, macro_conditions: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic borrower features.
        
        Args:
            noise: Random noise tensor [batch_size, noise_dim]
            macro_conditions: Macroeconomic conditions [batch_size, macro_dim]
            
        Returns:
            Generated borrower features [batch_size, borrower_dim]
        """
        # Concatenate noise and macro conditions
        input_tensor = torch.cat([noise, macro_conditions], dim=1)
        return self.model(input_tensor)


class Discriminator(nn.Module):
    """
    Discriminator network for conditional GAN.
    
    Distinguishes between real and fake borrower profiles given macro conditions.
    
    Args:
        borrower_dim: Dimension of borrower features
        macro_dim: Dimension of macroeconomic conditioning variables
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        borrower_dim: int = 15,
        macro_dim: int = 6,
        hidden_dims: list = [512, 256, 128]
    ):
        super(Discriminator, self).__init__()
        
        self.borrower_dim = borrower_dim
        self.macro_dim = macro_dim
        
        # Input dimension is borrower features + macro conditions
        input_dim = borrower_dim + macro_dim
        
        # Build discriminator layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer - probability of being real
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, borrower_features: torch.Tensor, macro_conditions: torch.Tensor) -> torch.Tensor:
        """
        Classify borrower profiles as real or fake.
        
        Args:
            borrower_features: Borrower feature tensor [batch_size, borrower_dim]
            macro_conditions: Macroeconomic conditions [batch_size, macro_dim]
            
        Returns:
            Probability of being real [batch_size, 1]
        """
        # Concatenate borrower features and macro conditions
        input_tensor = torch.cat([borrower_features, macro_conditions], dim=1)
        return self.model(input_tensor)


class ConditionalGAN:
    """
    Conditional GAN for credit risk stress testing.
    
    This class handles the training and generation process for creating
    synthetic borrower populations under stressed macroeconomic scenarios.
    """
    
    def __init__(
        self,
        noise_dim: int = 100,
        borrower_dim: int = 15,
        macro_dim: int = 6,
        generator_lr: float = 0.0002,
        discriminator_lr: float = 0.0002,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.noise_dim = noise_dim
        self.borrower_dim = borrower_dim
        self.macro_dim = macro_dim
        
        # Initialize networks
        self.generator = Generator(noise_dim, borrower_dim, macro_dim).to(self.device)
        self.discriminator = Discriminator(borrower_dim, macro_dim).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=generator_lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'epoch': []
        }
        
    def train_step(
        self, 
        real_borrowers: torch.Tensor, 
        macro_conditions: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Perform one training step.
        
        Args:
            real_borrowers: Real borrower features [batch_size, borrower_dim]
            macro_conditions: Macroeconomic conditions [batch_size, macro_dim]
            
        Returns:
            Tuple of (discriminator_loss, generator_loss)
        """
        batch_size = real_borrowers.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real data
        real_output = self.discriminator(real_borrowers, macro_conditions)
        d_real_loss = self.criterion(real_output, real_labels)
        
        # Fake data
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_borrowers = self.generator(noise, macro_conditions)
        fake_output = self.discriminator(fake_borrowers.detach(), macro_conditions)
        d_fake_loss = self.criterion(fake_output, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Generate fake data and try to fool discriminator
        fake_output = self.discriminator(fake_borrowers, macro_conditions)
        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def generate_synthetic_borrowers(
        self, 
        macro_conditions: torch.Tensor, 
        num_samples: int = 1000
    ) -> np.ndarray:
        """
        Generate synthetic borrower profiles for given macro conditions.
        
        Args:
            macro_conditions: Macroeconomic stress scenario [1, macro_dim] or [num_samples, macro_dim]
            num_samples: Number of synthetic borrowers to generate
            
        Returns:
            Generated borrower features [num_samples, borrower_dim]
        """
        self.generator.eval()
        
        with torch.no_grad():
            # Expand macro conditions if single scenario provided
            if macro_conditions.size(0) == 1:
                macro_conditions = macro_conditions.repeat(num_samples, 1)
            
            # Generate noise
            noise = torch.randn(num_samples, self.noise_dim).to(self.device)
            
            # Generate synthetic borrowers
            synthetic_borrowers = self.generator(noise, macro_conditions)
            
        return synthetic_borrowers.cpu().numpy()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'training_history': self.training_history,
            'config': {
                'noise_dim': self.noise_dim,
                'borrower_dim': self.borrower_dim,
                'macro_dim': self.macro_dim
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {filepath}")

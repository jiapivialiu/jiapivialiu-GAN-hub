#!/usr/bin/env python3
"""
Simple Test Script for Toy Training

Quick script to test the toy training functionality.
"""

import subprocess
import sys
import os

def test_toy_training():
    """Test the toy training script."""
    print("🧪 TESTING TOY TRAINING SCRIPT")
    print("="*40)
    
    # Check if we're in the right directory
    if not os.path.exists('src/models/conditional_gan.py'):
        print("❌ Error: Please run this from the GAN-for-stress-testing directory")
        return False
    
    # Check for the correct Python environment
    python_path = sys.executable
    if os.path.exists('../gan-env/bin/python'):
        python_path = '../gan-env/bin/python'
        print(f"✅ Using gan-env Python: {python_path}")
    else:
        print(f"⚠️ Using system Python: {python_path}")
    
    # Test with minimal parameters
    cmd = [
        python_path, 
        'toy_training_debug.py',
        '--epochs', '5',
        '--samples', '500', 
        '--verbose'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Toy training completed successfully!")
            print("\n📊 Output:")
            print(result.stdout)
            return True
        else:
            print("❌ Toy training failed!")
            print("\n🐛 Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running toy training: {e}")
        return False

if __name__ == "__main__":
    success = test_toy_training()
    sys.exit(0 if success else 1)

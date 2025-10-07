#!/usr/bin/env python3
"""
Quick verification script to test the GAN stress testing environment setup.
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported successfully."""
    try:
        if package_name:
            module = importlib.import_module(module_name, package_name)
        else:
            module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {module_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: Import failed - {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 GAN Stress Testing Environment Verification")
    print("=" * 50)
    
    # Python version
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print()
    
    # Test core dependencies
    print("Testing Core Dependencies:")
    print("-" * 30)
    
    success_count = 0
    total_count = 0
    
    dependencies = [
        'torch',
        'torchvision', 
        'numpy',
        'pandas',
        'sklearn',
        'scipy',
        'matplotlib',
        'seaborn',
        'yaml',
        'jupyter',
        'plotly',
        'tqdm'
    ]
    
    for dep in dependencies:
        total_count += 1
        if test_import(dep):
            success_count += 1
    
    print()
    print("Summary:")
    print("-" * 10)
    print(f"Successful imports: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All dependencies are correctly installed!")
        print("✅ Environment is ready for GAN stress testing!")
    else:
        print("⚠️ Some dependencies are missing. Please check the failed imports above.")
    
    # Test PyTorch GPU availability
    print()
    print("PyTorch Configuration:")
    print("-" * 20)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
        else:
            print("Running on CPU (CUDA not available)")
    except Exception as e:
        print(f"Error checking PyTorch: {e}")

if __name__ == "__main__":
    main()

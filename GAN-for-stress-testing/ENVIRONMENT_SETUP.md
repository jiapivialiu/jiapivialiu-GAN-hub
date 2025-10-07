# Virtual Environment Setup Guide

This guide explains how to set up and use the virtual environment for the GAN-based stress testing project.

## Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
cd /Users/jiapivialiu/Documents/Study/Github_repos/jiapivialiu-GAN-hub/GAN-for-stress-testing
./setup_environment.sh
```

### Option 2: Manual Setup
```bash
# 1. Navigate to project directory
cd /Users/jiapivialiu/Documents/Study/Github_repos/jiapivialiu-GAN-hub/GAN-for-stress-testing

# 2. Create virtual environment
python -m venv gan-stress-env

# 3. Activate virtual environment
source gan-stress-env/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python verify_environment.py
```

## 📦 Environment Details

- **Python Version**: 3.13.2
- **Virtual Environment**: `gan-stress-env`
- **Location**: `/Users/jiapivialiu/Documents/Study/Github_repos/jiapivialiu-GAN-hub/GAN-for-stress-testing/gan-stress-env`

## 🔧 Installed Packages

### Core ML/AI Libraries
- **PyTorch**: 2.8.0 (Deep learning framework)
- **torchvision**: 0.23.0 (Computer vision utilities)
- **scikit-learn**: 1.7.2 (Machine learning)
- **numpy**: 2.3.3 (Numerical computing)
- **pandas**: 2.3.3 (Data manipulation)
- **scipy**: 1.16.2 (Scientific computing)

### Visualization & Analysis
- **matplotlib**: 3.10.6 (Plotting)
- **seaborn**: 0.13.2 (Statistical visualization)
- **plotly**: 6.3.1 (Interactive plots)

### Jupyter & Notebooks
- **jupyter**: 1.1.1 (Jupyter ecosystem)
- **notebook**: 7.4.5 (Jupyter notebooks)
- **ipywidgets**: 8.1.7 (Interactive widgets)

### Utilities
- **PyYAML**: 6.0.2 (Configuration files)
- **tqdm**: 4.67.1 (Progress bars)

## 🎯 Usage Instructions

### Activating the Environment
```bash
cd /Users/jiapivialiu/Documents/Study/Github_repos/jiapivialiu-GAN-hub/GAN-for-stress-testing
source gan-stress-env/bin/activate
```

### Running the Demo
```bash
# Start Jupyter notebook
jupyter notebook notebooks/demo_stress_testing.ipynb

# Or run verification
python verify_environment.py

# Or test modules
python test_modules.py
```

### Deactivating the Environment
```bash
deactivate
```

## 🔍 Verification

The environment includes several verification scripts:

1. **verify_environment.py**: Tests all dependencies and PyTorch configuration
2. **test_modules.py**: Tests custom GAN stress testing modules
3. **setup_environment.sh**: Automated setup and verification

## Environment Status

**Virtual Environment**: Created and activated  
**Dependencies**: All packages installed successfully  
**PyTorch**: Working (CPU mode)  
**Custom Modules**: All modules importing correctly  
**Jupyter**: Configured and ready  

## Troubleshooting

### If imports fail:
```bash
# Ensure you're in the virtual environment
which python
# Should show: .../gan-stress-env/bin/python

# Reinstall packages if needed
pip install -r requirements.txt
```

### If PyTorch GPU not available:
- This is normal on macOS without NVIDIA GPU
- The system will run on CPU, which is sufficient for the demo

### If notebook kernel not found:
```bash
# Install ipykernel in the virtual environment
pip install ipykernel
python -m ipykernel install --user --name=gan-stress-env
```

## 📁 Project Structure

```
GAN-for-stress-testing/
├── gan-stress-env/          # Virtual environment
├── src/                     # Source code
├── notebooks/               # Jupyter notebooks
├── configs/                 # Configuration files
├── requirements.txt         # Python dependencies
├── setup_environment.sh     # Setup script
├── verify_environment.py    # Verification script
└── test_modules.py         # Module testing
```

## Next Steps

1. **Run Demo**: Start with `jupyter notebook notebooks/demo_stress_testing.ipynb`
2. **Explore Code**: Browse the `src/` directory for implementation details
3. **Customize**: Modify `configs/default_config.yaml` for your scenarios
4. **Extend**: Add your own data and scenarios

The environment is now ready for GAN-based credit risk stress testing!

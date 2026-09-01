# DeGibbs Virtual Environment Setup
# Run this script from the project root directory

# Create virtual environment
Write-Host "Creating Python virtual environment..." -ForegroundColor Green
python -m venv .venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version)
# For CUDA 11.8:
Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Green
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only, uncomment below:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the package in development mode
Write-Host "Installing DeGibbs package..." -ForegroundColor Green
pip install -e .

# Install development dependencies
Write-Host "Installing development dependencies..." -ForegroundColor Green
pip install -e ".[dev]"

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "To activate the environment, run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "To train, run: python train.py --data_dir /path/to/imagenet --exp_dir ./experiments" -ForegroundColor Yellow

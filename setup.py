"""Setup script for DeGibbs package."""

from setuptools import setup, find_packages

setup(
    name="dldegibbs",
    version="2.0.0",
    description="Deep Learning for Gibbs Artifact Removal in MRI",
    author="NYU School of Medicine / Modified for 129Xe MRI",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "Pillow>=9.4.0",
        "PyWavelets>=1.4.0",
        "PyYAML>=6.0",
        "tensorboard>=2.12.0",
        "rich>=13.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "degibbs-train=train:main",
        ],
    },
)

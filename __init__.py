"""
Counting Flows: Count-based Flow Matching with GPU Bridges

A clean, Hydra-based implementation of count-based flow matching using GPU-accelerated
bridges for efficient training and sampling.

Main Components:
- GPU Bridges (CuPy-based): Fast bridge operations on GPU
- Simplified Datasets: Mixture of Poissons with sampled parameters
- Hydra Configuration: Structured configuration management
- Clean Training Loop: Streamlined training with proper GPU integration

Usage:
    from counting_flows import main_hydra
    
    # Or run directly:
    python run.py
    python run.py --config-name=experiment_quick
"""

# Version info
__version__ = "2.0.0"
__author__ = "Counting Flows Team"
__description__ = "Count-based Flow Matching with GPU Bridges"

# Main components for easy import
__all__ = [
    # Core components
    # Main function
    "main_hydra",
]

# Optional import of main function
try:
    from .main_hydra import main as main_hydra
    __all__.append("main_hydra")
except ImportError:
    # Hydra might not be available in all contexts
    pass 
"""
Command Line Interface for Count-based Flow Matching

Provides argument parsing and configuration for the counting flows framework.
"""

import argparse


def parse_args():
    """Parse command line arguments for count-based flow matching"""
    parser = argparse.ArgumentParser(description="Count-based Flow Matching")
    
    # Dataset selection
    parser.add_argument(
        "--dataset", "--data",
        choices=["poisson", "betabinomial"],
        default="poisson",
        help="Dataset type: 'poisson' (Poisson endpoints), 'betabinomial' (BetaBinomial endpoints with small alpha/beta)"
    )
    
    # Data configuration
    parser.add_argument(
        "--data-dim", "--dim", "-d", type=int, default=4,
        help="Dimensionality of count vectors"
    )
    
    parser.add_argument(
        "--fixed-base", action="store_true",
        help="Use fixed base measure (x₀) instead of random"
    )
    
    # Model architecture
    parser.add_argument(
        "--arch", "--architecture", 
        choices=["nb", "mle", "bb", "zip"], 
        default="nb",
        help="Neural architecture: 'nb' (Negative Binomial), 'mle' (MLE Regressor), 'bb' (Beta-Binomial), or 'zip' (Zero-Inflated Poisson)"
    )
    
    # Bridge mode
    parser.add_argument(
        "--bridge", "--bridge-mode",
        choices=["nb", "poisson", "poisson_bd", "polya_bd"],
        default="nb", 
        help="Bridge type: 'nb' (Polya/Beta-Binomial), 'poisson' (exact Poisson), 'poisson_bd' (Poisson Birth-Death), or 'polya_bd' (Polya Birth-Death)"
    )
    
    # Sampling bridge mode (can be different from training)
    parser.add_argument(
        "--sample-bridge", 
        choices=["nb", "poisson", "poisson_bd", "polya_bd", "auto"],
        default="auto",
        help="Bridge type for sampling (auto=same as training bridge)"
    )
    
    # NEW: Scheduling options
    parser.add_argument(
        "--r-schedule", 
        choices=["linear", "cosine", "exponential", "sigmoid", "polynomial", "sqrt", "inverse_sqrt"],
        default="linear",
        help="Schedule for r(t) parameter in NB bridge"
    )
    
    parser.add_argument(
        "--time-schedule", 
        choices=["uniform", "early_dense", "late_dense", "middle_dense"],
        default="uniform",
        help="Time point distribution for bridge sampling"
    )
    
    parser.add_argument(
        "--sample-r-schedule", 
        choices=["linear", "cosine", "exponential", "sigmoid", "polynomial", "sqrt", "inverse_sqrt", "auto"],
        default="auto",
        help="Schedule for r(t) during sampling (auto=same as training)"
    )
    
    parser.add_argument(
        "--sample-time-schedule", 
        choices=["uniform", "early_dense", "late_dense", "middle_dense", "auto"],
        default="auto",
        help="Time schedule for reverse sampling (auto=same as training)"
    )
    
    # Schedule-specific parameters
    parser.add_argument("--decay-rate", type=float, default=2.0, help="Decay rate for exponential schedule")
    parser.add_argument("--steepness", type=float, default=10.0, help="Steepness for sigmoid schedule")
    parser.add_argument("--midpoint", type=float, default=0.5, help="Midpoint for sigmoid schedule")
    parser.add_argument("--power", type=float, default=2.0, help="Power for polynomial schedule")
    parser.add_argument("--concentration", type=float, default=2.0, help="Concentration for time spacing schedules")
    
    # Training hyperparameters
    parser.add_argument("--lr", "--learning-rate", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--batch-size", "-B", type=int, default=128, help="Batch size")
    parser.add_argument("--steps", "--n-steps", type=int, default=30, help="Number of diffusion steps")
    parser.add_argument("--iterations", "-i", type=int, default=20000, help="Training iterations")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer size")
    
    # DataLoader options
    parser.add_argument("--dataset-size", type=int, default=None, 
                       help="Dataset size per epoch (default: max(iterations, 10000))")
    parser.add_argument("--num-workers", type=int, default=0, 
                       help="Number of DataLoader workers (0=single-threaded)")
    parser.add_argument("--pin-memory", action="store_true", default=True,
                       help="Use pinned memory for faster GPU transfers")
    
    # Schedule parameters
    parser.add_argument("--r-min", type=float, default=1.0, help="Minimum r value for NB schedule")
    parser.add_argument("--r-max", type=float, default=20.0, help="Maximum r value for NB schedule")
    
    # Birth-death bridge parameters
    parser.add_argument("--bd-r", type=float, default=1.0, help="r parameter for Polya birth-death bridge")
    parser.add_argument("--bd-beta", type=float, default=1.0, help="beta parameter for Polya birth-death bridge")
    parser.add_argument("--lam-p0", type=float, default=8.0, help="Birth rate at t=0 for BD bridges")
    parser.add_argument("--lam-p1", type=float, default=8.0, help="Birth rate at t=1 for BD bridges")
    parser.add_argument("--lam-m0", type=float, default=8.0, help="Death rate at t=0 for BD bridges")
    parser.add_argument("--lam-m1", type=float, default=8.0, help="Death rate at t=1 for BD bridges")
    parser.add_argument("--bd-schedule", choices=["constant", "linear", "cosine"], default="constant",
                       help="Schedule type for birth/death rates in BD bridges")
    
    # Generation parameters
    parser.add_argument("--gen-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--use-mean", action="store_true", help="Use mean instead of sampling during generation")
    
    # Debugging
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plots and visualizations")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer iterations")
    
    # Device
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto", help="Device to use")
    
    return parser.parse_args() 
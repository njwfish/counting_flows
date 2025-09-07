# Multimodal Count Bridges: A Unified Framework for Discrete Diffusion

A research implementation of **multimodal count bridges** that combines continuous flow matching, discrete flow matching, and novel count diffusion models. This framework enables sophisticated deconvolution modeling where only aggregated observations are available, with applications to spatial transcriptomics, bulk RNA-seq deconvolution, and multimodal generation tasks.

## 🔬 Research Overview

This repository implements the methods from our paper on **discrete bridges for count diffusion**. The core innovation is a Poisson birth-death bridge that enables exact reverse-time generation with Chapman-Kolmogorov consistency and projectivity properties. The framework extends to:

- **Multimodal generation**: Joint modeling of images and count vectors
- **Deconvolution modeling**: Learning unit-level distributions from aggregate observations 
- **Conditional generation**: Incorporating side information and constraints
- **Optimal transport connections**: Theoretical links to Schrödinger bridges

### Key Mathematical Contributions

1. **Exact Discrete Bridge**: A Poisson birth-death process with exact reverse kernels that satisfy Chapman-Kolmogorov and projectivity
2. **Distributional Scoring**: Energy score and CRPS losses for modeling count distributions rather than point estimates
3. **Aggregate Deconvolution**: Generalized EM framework for learning from partial observations
4. **Multimodal Integration**: Joint diffusion across continuous and discrete modalities

## 🎯 Core Methods

### Poisson Birth-Death Bridge

The fundamental building block is a discrete bridge process that decomposes jumps into:
- **Backbone**: Directed jumps matching the endpoint gap `d₁ = x₁ - x₀`
- **Slack pairs**: Canceling `(+1, -1)` events that create variance

```python
# Forward process: condition on endpoints (x₀, x₁)
M₁ ~ Bessel(|d₁|; Λ₊(1), Λ₋(1))  # Slack pairs
N₁ = |d₁| + 2M₁                   # Total jumps
B₁ = (N₁ + d₁) / 2                # Positive jumps

# Reverse kernel t → s with exact Chapman-Kolmogorov
N_s | N_t ~ Binomial(N_t, W(s)/W(t))
B_s | (N_t, N_s, B_t) ~ Hypergeometric(N_t, B_t, N_s)
X_s = x₀ + 2B_s - N_s
```

### Deconvolution with Aggregate Constraints

When observing only aggregates `a₀ = Σ_g x_{g0}`, we use a generalized EM approach:

**E-Step**: Sample from aggregate-conditional posterior using guided diffusion:
1. Start from observed `x₁` 
2. At each reverse step: predict `x̂₀`, project to satisfy aggregate constraint
3. Use projected endpoint for next reverse step

**M-Step**: Train on distributional score with aggregate supervision:
```python
# Aggregate energy score
S_agg(Q_θ(·|x_t,t,z), a₀) = ½𝔼[ρ(A(X),A(X'))] - 𝔼[ρ(A(X),a₀)]
```

### Multimodal Architecture

Joint modeling of images and counts using U-ViT transformer:
- **Image patches**: Standard patch embedding for continuous data
- **Count patches**: Learned embedding that respects discrete structure  
- **Unified processing**: Single transformer backbone with modality-specific decoders
- **Cross-modal conditioning**: Images condition count generation and vice versa

## 🚀 Quick Start

### Basic Count Diffusion
```bash
# Train discrete count bridge on Poisson mixtures
python main.py bridge=skellam model=energy_score dataset=gaussian_mixture

# Constrained generation with aggregate targets
python main.py bridge=constrained model=energy_score_deconv
```

### Deconvolution Modeling
```bash
# Learn unit-level from aggregates using deconvolution
python main.py --config-name=config_deconv \
  model=energy_score_deconv \
  dataset=gaussian_deconv \
  training=deconv

# Multimodal deconvolution (images + counts)
python main.py --config-name=multimodal_lowrank_deconv \
  architecture=uvit_multimodal \
  model=energy_multimodal_deconv
```

### Multimodal Generation
```bash
# Joint image-count generation
python main.py --config-name=multimodal_lowrank \
  architecture=uvit_multimodal \
  model=energy_multimodal \
  dataset=mnist_lowrank_gaussian_mixture

# With genomic sequence conditioning
python main.py --config-name=expr \
  dataset=exprbyseq \
  architecture=scformer
```

## 📊 Applications

### Spatial Transcriptomics Deconvolution
```python
# Learn cell-level expression from spatial aggregates
from datasets.exprbyseq import ExprBySeq
from models.energy_deconv import DeconvolutionEnergyScoreLoss

dataset = ExprBySeq(
    snapfile="spatial_data.h5ad",
    window_size=1000,
    batch_size=100
)

model = DeconvolutionEnergyScoreLoss(
    architecture=architecture,
    noise_dim=16,
    m_samples=16
)
```

### Bulk RNA-seq Deconvolution
Decompose bulk measurements into cell-type specific expression while modeling uncertainty in the allocation.

### Multimodal Single-cell Analysis
Joint generation of cell images and gene expression profiles with cross-modal consistency.

## 🏗️ Architecture

The framework is built around modular components:

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│     Bridges     │    │   Architectures  │    │      Models      │
│                 │    │                  │    │                  │
│ • Skellam       │    │ • MLP            │    │ • Energy Score   │
│ • Constrained   │    │ • U-ViT          │    │ • Energy Deconv  │
│ • Multimodal    │    │ • SCFormer       │    │ • Multimodal     │
│ • CFM/DFM       │    │ • Attention      │    │ • Cross-Entropy  │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                │
                       ┌──────────────────┐
                       │     Datasets     │
                       │                  │
                       │ • Gaussian Mix   │
                       │ • MNIST+Counts   │
                       │ • ExprBySeq      │
                       │ • Deconvolution  │
                       └──────────────────┘
```

### Key Components

**Bridges** (`bridges/`):
- **Count Bridges**: `cupy/skellam.py`, `numpy/skellam.py` - GPU/CPU Poisson birth-death bridges
- **Constrained Bridges**: `cupy/constrained.py`, `numpy/constrained.py` - Mean-constrained count bridges 
- **Baseline Bridges**: `torch/cfm.py` (Continuous Flow Matching), `torch/dfm.py` (Discrete Flow Matching)
- **Diffusion Bridge**: `torch/diffusion.py` - VPSDE diffusion implementation
- **Multimodal Bridge**: `torch/multimodal.py` - Wrapper for joint multi-modality modeling

**Models/Losses** (`models/`):
- `energy.py`: Core distributional energy score loss
- `energy_deconv.py`: Deconvolution with aggregate constraints and KL projection
- `energy_multimodal.py`: Joint image-count energy scoring
- `energy_multimodal_deconv.py`: Combined multimodal + deconvolution
- `mse.py`, `cross_entropy.py`, `dfm.py`: Baseline losses for comparison

**Architectures** (`architecture/`):
- `mlp.py`: Flexible MLPs with configurable input concatenation
- `attention.py`: BERT-style transformer with positional embeddings
- `pos_unet.py`: Positional U-Net with encoder-decoder structure
- `uvit.py`, `uvit_multimodal.py`: U-ViT transformers for (multimodal) generation
- `scformer.py`: Transformer for genomic sequence conditioning

**Datasets** (`datasets/`):
- `gaussian_mixture.py`: Gaussian mixtures with integer outputs
- `gaussian_deconv.py`: Deconvolution benchmarks with aggregate constraints
- `mnist.py`, `mnist_mixture.py`: MNIST and MNIST+count multimodal datasets
- `exprbyseq.py`: Spatial transcriptomics with genomic sequence conditioning
- `discrete_moons.py`: Discrete version of two-moons dataset

## 🔧 Installation

```bash
git clone https://github.com/njwfish/counting_flows.git
cd counting_flows

# Install dependencies
pip install -r requirements.txt

# GPU acceleration (optional but recommended)
pip install cupy-cuda12x  # For CUDA 12.x
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CuPy (for GPU acceleration)
- Hydra (for configuration management)
- NumPy, SciPy, scikit-learn

## 📚 Theoretical Background

### Connection to Schrödinger Bridges

The discrete bridge exhibits the same phase transition as continuous Schrödinger bridges:
- **High noise** (`κ → ∞`): Approaches independence (maximum entropy)
- **Low noise** (`κ → 0`): Approaches optimal transport (minimum cost)

### Learning Guarantees

Under realizability and recoverability assumptions, adaptive training with aggregate supervision provides gradient approximations that converge to the oracle unit-level gradients. The approximation quality depends on the best aggregate prediction achievable across all time points.

### Distributional Scoring

Energy score and CRPS losses are strictly proper scoring rules that capture full distributional information rather than just point estimates. This enables uncertainty quantification in count predictions.

## 🎛️ Configuration

The framework uses Hydra for structured configuration management:

```yaml
# config.yaml (default configuration)
defaults:
  - bridge: skellam                              # Count bridge type
  - model: energy_score                          # Distributional loss
  - architecture: mlp                            # Neural architecture  
  - architecture/in_dims: with_noise             # Input dimension spec
  - architecture/out_dim: scalar                 # Output dimension spec
  - dataset: low_rank_gaussian_mixture_5d        # Data source
  - training: default                            # Training parameters
  - slack_sampler: bessel                        # Slack sampling method

architecture:
  act_fn: softplus                               # Activation for count outputs

device: cuda
seed: 42
train_split: 0.8
```

### Available Configuration Options

**Bridges**: `skellam`, `constrained`, `cfm`, `dfm`, `diffusion`, `multimodal`
**Models**: `energy_score`, `energy_deconv`, `energy_multimodal`, `energy_multimodal_deconv`, `mse`, `cross_entropy`, `dfm`  
**Architectures**: `mlp`, `attention`, `pos_unet`, `uvit`, `uvit_multimodal`, `scformer`
**Datasets**: `gaussian_mixture`, `gaussian_deconv`, `mnist_mixture`, `exprbyseq`, `discrete_moons`

### Advanced Configurations

**Deconvolution setup**:
```yaml
# config_deconv.yaml
model: energy_score_deconv
dataset: gaussian_deconv
training: deconv
```

**Multimodal setup**:
```yaml  
# multimodal_lowrank.yaml
architecture: uvit_multimodal
model: energy_multimodal
bridge: multimodal
dataset: mnist_lowrank_gaussian_mixture
```

**Input/Output Dimension Flexibility**:
```yaml
# architecture/in_dims/with_M_t_and_noise.yaml - includes slack variables
# architecture/in_dims/with_context.yaml - includes conditioning
# architecture/out_dim/discrete.yaml - for categorical outputs
```

## 🔧 Implementation Highlights

### GPU-Accelerated Count Bridges
The repository includes both CuPy (`bridges/cupy/`) and NumPy (`bridges/numpy/`) implementations of the count bridges, enabling efficient GPU computation for large-scale problems.

### Flexible Architecture System
The modular architecture system supports:
- **Input concatenation**: Automatically handles different input types (time, noise, context, slack variables)
- **Configurable dimensions**: List-based input/output specifications via Hydra configs
- **Multi-output support**: Single models can output both images and count vectors

### Sophisticated Sampling
- **Slack samplers**: Multiple implementations (`bessel`, `poisson`, `const`) for different slack variable distributions
- **Randomized rounding**: Exact integer projection algorithms for deconvolution constraints
- **Trajectory optimization**: Adaptive endpoint selection for stable deconvolution training

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
- `skellam.py`: Core Poisson birth-death bridge implementation
- `constrained.py`: Mean-constrained bridges for aggregate modeling
- `multimodal.py`: Joint bridges for mixed data types

**Models** (`models/`):
- `energy_score.py`: Distributional energy score loss
- `energy_deconv.py`: Deconvolution with aggregate constraints
- `energy_multimodal.py`: Joint image-count modeling

**Architectures** (`architecture/`):
- `uvit_multimodal.py`: Multimodal U-ViT for joint generation
- `scformer.py`: Transformer for genomic sequence conditioning
- `mlp.py`: Flexible MLPs with configurable inputs/outputs

**Datasets** (`datasets/`):
- `exprbyseq.py`: Spatial transcriptomics with sequence conditioning
- `gaussian_deconv.py`: Synthetic deconvolution benchmarks
- `mnist_mixture.py`: Multimodal image-count datasets

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
# config.yaml
defaults:
  - bridge: skellam          # Discrete bridge type
  - model: energy_score      # Distributional loss
  - architecture: uvit       # Neural architecture  
  - dataset: gaussian_mixture # Data source
  - training: default        # Training parameters

data_dim: 10
n_steps: 50
training:
  num_epochs: 1000
  batch_size: 512
```

### Advanced Configurations

**Deconvolution setup**:
```yaml
# config_deconv.yaml
model: energy_score_deconv
dataset: gaussian_deconv
training: deconv
group_size: 10
agg_noise: 0.1
```

**Multimodal setup**:
```yaml
# multimodal_lowrank.yaml
architecture: uvit_multimodal
model: energy_multimodal
bridge: multimodal
dataset: mnist_lowrank_gaussian_mixture
```

## 📈 Experimental Results

The framework has been validated on:

1. **Synthetic count generation**: Poisson mixtures with exact bridge validation
2. **Deconvolution benchmarks**: Recovery of unit-level structure from aggregates  
3. **Spatial transcriptomics**: Cell-level expression from spatial measurements
4. **Multimodal generation**: Joint image-count modeling with cross-modal consistency

Key findings:
- Exact Chapman-Kolmogorov property enables stable training
- Distributional losses outperform MSE for count uncertainty
- Deconvolution works best with moderate group sizes (10-100 units)
- Multimodal architectures capture cross-modal dependencies

## 🤝 Contributing

This is a research codebase under active development. Contributions are welcome, particularly:
- New bridge implementations for different discrete distributions
- Additional baseline methods for comparison
- Applications to new domains (chemistry, ecology, etc.)
- Theoretical extensions and analysis

## 📋 License

MIT License - see `LICENSE` file for details.

## 🔗 Related Work

- **Flow Matching**: Continuous normalizing flows with optimal transport
- **Discrete Diffusion**: Categorical and ordinal diffusion models  
- **Deconvolution**: Bulk RNA-seq and spatial transcriptomics analysis
- **Schrödinger Bridges**: Optimal transport with entropic regularization
- **Count Data Models**: Poisson regression and negative binomial models
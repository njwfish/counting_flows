# Counting Flow Matching Framework

A modular, Hydra-based framework for **count-based flow matching** with specialized support for discrete count data. Features novel Skellam bridges, energy score losses, and flexible architectures, with additional baseline methods (CFM, DFM) for comparison.

## ✨ Key Features

- **🔢 Count-Based Flow Matching**: Novel Skellam birth-death bridges for non-negative integer count data
- **⚡ GPU-Accelerated Bridges**: CuPy-based implementations for fast count diffusion processes  
- **📊 Distributional Losses**: Energy Score and CRPS losses designed for count distributions
- **🧠 Flexible Architectures**: MLP and attention-based architectures with configurable input/output dimensions
- **📈 Specialized Datasets**: Poisson mixtures and count-based transformations
- **🔄 Baseline Methods**: CFM and Discrete Flow Matching for comparison and ablation studies
- **⚙️ Modular Design**: Clean, composable components that work together seamlessly
- **🔧 Hydra Configuration**: Structured, reproducible configuration management
- **💾 Smart Checkpointing**: Automatic training resumption and model saving
- **🎯 Easy Extensibility**: Add new components with minimal code changes

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/njwfish/counting_flows.git
cd counting_flows

# Install dependencies
pip install -r requirements.txt
pip install cupy-cuda12x  # For CUDA 12.x
```

## 🔬 Novel Contributions

This framework introduces **count-based flow matching**, a new paradigm for modeling discrete count data:

- **🎯 Specialized for Counts**: Unlike standard flow matching that treats discrete data as continuous or categorical, our approach respects the non-negative integer nature of count data
- **⚡ Skellam Bridges**: Novel use of birth-death processes with time-varying rates for natural count transformations
- **📊 Distributional Modeling**: Energy Score and CRPS losses that capture uncertainty in count predictions, not just point estimates
- **🧮 GPU Acceleration**: CuPy-based implementations for efficient count diffusion on GPUs
- **🎚️ Constrained Generation**: Mean-constrained bridges for controlled count generation

**Why Count Flows Matter**: Count data appears everywhere (word frequencies, neural spike trains, reaction counts, etc.) but existing flow methods either ignore the discrete constraint or treat counts as arbitrary categories. Count flows naturally handle the structure of non-negative integers while modeling full distributions.

## 🚀 Quick Start

### Basic Usage

```bash
# Train with default configuration (count-based flow matching)
python main.py

# Train with attention architecture for complex count dependencies
python main.py architecture=attention_discrete

# Quick experimentation with different count parameters
python main.py data_dim=8 n_steps=50 training.num_epochs=200

# Baseline methods for comparison
python main.py bridge=cfm --config-name=cfm_config     # Continuous Flow Matching
python main.py --config-name=dfm_config dataset=discrete_moons  # Discrete Flow Matching
```

### Configuration Examples

```bash
# Count-based flow matching (main contribution)
python main.py bridge=skellam model=energy_score      # Standard count flows
python main.py bridge=constrained model=energy_score  # Constrained count flows

# Different architectures for count data
python main.py architecture=mlp               # Standard MLP (fast)
python main.py architecture=attention_discrete # Transformer-based attention (complex dependencies)

# Baseline comparison methods
python main.py bridge=cfm model=mse              # Continuous Flow Matching
python main.py bridge=dfm model=cross_entropy    # Discrete Flow Matching
```

## 🏗️ Architecture Overview

The framework is built around four core components that can be mixed and matched:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dataset   │ -> │   Bridge    │ -> │    Model    │ -> │  Training   │
│             │    │             │    │             │    │             │
│ • Poisson   │    │ • CFM       │    │ • MSE       │    │ • Adam      │
│ • Discrete  │    │ • DFM       │    │ • Energy    │    │ • Logging   │
│   Moons     │    │ • Skellam   │    │ • CRPS      │    │ • Checkpts  │
│             │    │             │    │ • CrossEnt  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                      ┌─────────────┐
                                      │    Arch     │
                                      │             │
                                      │ • MLP       │
                                      │ • Attention │
                                      └─────────────┘
```

## 📚 Components

### 🌊 Bridges (`bridges/`)

**Count-Based Flows (Main Contribution):**
- **Skellam** (`bridges/cupy/skellam.py`): Novel Skellam birth-death processes for count data
- **Constrained Skellam** (`bridges/cupy/constrained.py`): Mean-constrained count flows

**Baseline Methods:**
- **CFM** (`bridges/baselines/cfm.py`): Continuous Flow Matching with optimal transport
- **DFM** (`bridges/baselines/dfm.py`): Discrete Flow Matching with zero masking

### 🧠 Architectures (`architecture/`)

**MLP** (`architecture/mlp.py`):
- Flexible input/output dimensions via lists
- Supports arbitrary keyword arguments
- Automatic input concatenation

**Attention** (`architecture/attention.py`):  
- BERT-like transformer architecture
- Intelligent input handling (splits vs broadcasts)
- Learnable position embeddings
- Multi-head self-attention

### 🎯 Models (`models/`)

**Count-Based Models (Main Contribution):**
- **Energy Score** (`models/energy.py`): Distributional energy score with m-sample approximation for count distributions
- **CRPS** (`models/crps.py`): Continuous Ranked Probability Score for distributional prediction

**Baseline Models:**
- **MSE** (`models/mse.py`): Simple mean squared error loss for continuous data
- **Cross-Entropy** (`models/cross_entropy.py`): For discrete flow matching with categorical distributions

### 📊 Datasets (`datasets/`)

**Count Data (Main Focus):**
- **Poisson Mixture** (`datasets/poisson_mixture.py`): Mixture of Poisson distributions for count data with pre-sampling for efficiency and configurable mixture components

**Baseline Datasets:**
- **Discrete Moons** (`datasets/discrete_moons.py`): 8-gaussians → 2-moons discrete flow task, integerized using consistent scaling

## ⚙️ Configuration System

### Directory Structure

```
configs/
├── config.yaml              # Default continuous flow config
├── dfm_config.yaml          # Discrete flow matching config  
├── architecture/
│   ├── mlp.yaml             # Standard MLP
│   ├── discrete_attention.yaml # Attention for discrete flows
│   └── attention.yaml       # General attention config
├── bridge/
│   ├── cfm.yaml            # Continuous Flow Matching
│   ├── dfm.yaml            # Discrete Flow Matching
│   └── skellam.yaml        # Skellam bridge
├── model/
│   ├── mse.yaml            # MSE loss
│   ├── energy_score.yaml  # Energy score loss
│   └── cross_entropy.yaml # Cross-entropy loss
├── dataset/
│   ├── poisson_mixture.yaml # Poisson mixtures
│   └── discrete_moons.yaml  # 8-gaussians→2-moons
└── training/
    └── default.yaml         # Training hyperparameters
```

### Key Configuration Patterns

**Flexible Input/Output Dimensions:**
```yaml
# MLP with list inputs/outputs
in_dims: 
  - ${data_dim}  # x_t
  - 1            # time
out_dim:
  - ${data_dim}  # output dimensions
  - 128          # vocab size (for discrete)
```

**Component Composition:**
```yaml
defaults:
  - bridge: dfm                    # Choose bridge type
  - model: cross_entropy          # Choose loss function
  - architecture: discrete_attention # Choose architecture
  - dataset: discrete_moons       # Choose dataset
```

## 🔄 Flow Types Explained

### Count-Based Flow Matching (Main Contribution)

**Novel approach** for count data using birth-death processes with Skellam bridges:
- **Innovation**: Specialized jump processes for non-negative integer count data
- **Bridge**: `skellam` - Stochastic birth-death processes with time-varying rates  
- **Models**: `energy_score`, `crps` - Distributional losses designed for count distributions
- **Data**: Non-negative integer counts (e.g., word frequencies, spike counts, reaction counts)
- **Advantages**: Naturally handles discrete, non-negative constraints and distributional uncertainty

```bash
# Standard count-based flow matching
python main.py bridge=skellam model=energy_score

# With mean constraints for controlled generation
python main.py bridge=constrained model=energy_score
```

### Baseline Comparison Methods

**Continuous Flow Matching (CFM)** - Standard approach for continuous data:
- **Bridge**: `cfm` - Optimal transport between distributions
- **Models**: `mse`, `energy_score` - Continuous loss functions  
- **Data**: Real-valued vectors

```bash
python main.py bridge=cfm model=energy_score --config-name=cfm_config
```

**Discrete Flow Matching (DFM)** - Categorical approach for discrete data:
- **Bridge**: `dfm` - Zero masking with proportional interpolation
- **Models**: `cross_entropy` - Categorical loss function
- **Data**: Integer-valued vectors (vocabulary-based)

```bash
python main.py --config-name=dfm_config dataset=discrete_moons
```

## 🛠️ Extending the Framework

### Adding a New Bridge

1. **Implement bridge** in `bridges/`:
```python
class MyBridge:
    def __call__(self, x_0, x_1, t_target=None):
        # Bridge logic here
        return {"inputs": {...}, "output": ...}
    
    def sampler(self, x_1, z, model, **kwargs):
        # Sampling logic here
        return samples
```

2. **Add configuration** in `configs/bridge/my_bridge.yaml`:
```yaml
_target_: bridges.my_bridge.MyBridge
param1: value1
param2: value2
```

### Adding a New Model

1. **Implement model** in `models/`:
```python
class MyModel(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
    
    def forward(self, inputs):
        return self.architecture(**inputs)
    
    def sample(self, **kwargs):
        return self.forward(kwargs)
    
    def loss(self, target, inputs):
        # Loss computation here
        return loss
```

2. **Add configuration** in `configs/model/my_model.yaml`:
```yaml
_target_: models.my_model.MyModel
architecture: ${architecture}
```

### Adding a New Architecture

1. **Implement architecture** in `architecture/`:
```python
class MyArch(nn.Module):
    def __init__(self, in_dims, hidden_dim, out_dim):
        # Support flexible list dimensions
        super().__init__()
        # Implementation here
    
    def forward(self, **kwargs):
        # Handle arbitrary keyword inputs
        return output
```

2. **Add configuration** in `configs/architecture/my_arch.yaml`:
```yaml
_target_: architecture.my_arch.MyArch
in_dims: [${data_dim}, 1]
out_dim: ${data_dim}
hidden_dim: 64
```

## 📋 Example Workflows

### Count-Based Flow Matching Experiments (Main Focus)

```bash
# Standard count flow experiment
python main.py bridge=skellam model=energy_score data_dim=8 n_steps=50

# Count flows with attention for complex dependencies  
python main.py bridge=skellam model=energy_score architecture=attention_discrete

# Constrained count generation with mean control
python main.py bridge=constrained model=energy_score data_dim=4

# Ablation study: different count-based models
python main.py model=energy_score bridge=skellam   # Distributional energy score
python main.py model=crps bridge=skellam          # CRPS for count distributions
```

### Architecture Comparison for Count Data

```bash
# Compare architectures on count data
python main.py model=energy_score architecture=mlp               # Fast MLP
python main.py model=energy_score architecture=attention_discrete # Complex dependencies
```

### Baseline Comparison Studies

```bash
# Compare flow types on similar data
python main.py bridge=skellam model=energy_score dataset=poisson_mixture     # Count-based (main)
python main.py bridge=cfm model=energy_score dataset=poisson_mixture         # Continuous baseline
python main.py --config-name=dfm_config dataset=discrete_moons               # Discrete baseline
```

## 🔍 Monitoring and Visualization

- **Automatic checkpointing**: Models saved in `outputs/<timestamp>/`
- **Loss tracking**: Training progress logged automatically
- **Visualization**: Automatic plot generation (when enabled)
- **Config logging**: Full configuration saved with results

## 📊 Performance Tips

1. **GPU Usage**: Enable CUDA with `device=cuda` 
2. **Batch Size**: Tune `training.batch_size` for your GPU memory
3. **Architecture**: Try attention for complex dependencies, MLP for speed
4. **Steps**: More `n_steps` = better quality but slower sampling
5. **Checkpointing**: Use `training.save_every` to save progress

## 🔧 Troubleshooting

**Configuration Errors:**
```bash
# Check config composition
python main.py --cfg job

# Validate config without training  
python main.py --cfg hydra
```

**Memory Issues:**
- Reduce `training.batch_size`
- Reduce `data_dim` or `n_steps`  
- Use smaller `hidden_dim` in architectures
- For CuPy bridges: Check GPU memory with `nvidia-smi`

**Count Flow Issues:**
- Ensure non-negative integer data for Skellam bridges
- Use `model=energy_score` or `model=crps` with count data
- Check that data dimensions match architecture configurations
- For constrained bridges: Verify mean constraint satisfaction

**Baseline Comparison Issues:**
- Use `dataset=discrete_moons` for discrete flow experiments
- Use `model=cross_entropy` with discrete flows (DFM)
- Ensure vocabulary size matches in architecture configs for DFM

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{counting_flow_matching_framework,
  title={Counting Flow Matching Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## 📝 License

[Add your license here]

---

**🚀 Ready to start experimenting with count flows?** Try `python main.py` for count-based flow matching or explore the baseline methods for comparison! 
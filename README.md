# ICL Benchmarking

Benchmarking the performance of different transformer architectures (quadratic vs linear attention) on in-context learning tasks.

## Project Structure

```
icl-benchmarking/
├── src/                    # Source code
│   ├── models/            # Model definitions
│   │   ├── quadratic_attention.py
│   │   └── linear_attention.py
│   ├── training/          # Training utilities
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── utils/             # Utility functions
│   │   ├── data_utils.py
│   │   └── helpers.py
│   └── plotting/          # Visualization
│       └── plots.py
├── experiments/           # Experiment scripts
│   ├── run_linear_attention.py
│   ├── run_quadratic_attention.py
│   └── validate_data.py
├── scripts/               # Utility scripts
│   └── generate_data.py
├── data/                  # Data directory
│   ├── isotropic_data.npz
│   └── anisotropic_data.npz
└── results/               # Experiment results (generated)
```

## Setup

1. Install dependencies:
```bash
pip install torch numpy matplotlib tqdm transformers
```

2. Generate data:
```bash
python scripts/generate_data.py
```

3. Validate data:
```bash
python experiments/validate_data.py
```

## Running Experiments

### Linear Attention
```bash
python experiments/run_linear_attention.py
```

### Quadratic Attention
```bash
python experiments/run_quadratic_attention.py
```

You can modify the configuration dictionaries in these scripts to run different experiments (e.g., different numbers of layers, learning rates, etc.).

## Usage Example

```python
import torch
from src.models import LinearAttentionICLModel, TransformerModel
from src.utils import load_data
from src.training import train_multiseed
from src.plotting import plot_multiseed_history

# Load data
train_xs, train_ys, test_xs, test_ys = load_data('data/isotropic_data.npz')

# Configure experiment
config = {
    'seeds': [42, 100, 7],
    'lr': 3e-4,
    'steps': 10000,
    'batch_size': 64,
    'model_params': {
        'n_dims': 5,
        'n_positions': 10,
        'n_embd': 256,
        'n_layer': 6,
        'n_head': 4,
    }
}

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
histories, models = train_multiseed(
    seeds=config['seeds'],
    model_class=LinearAttentionICLModel,
    train_xs=train_xs,
    train_ys=train_ys,
    test_xs=test_xs,
    test_ys=test_ys,
    config=config,
    device=device
)

# Plot results
plot_multiseed_history(6, histories, model_type="Linear")
```

## Models

- **Quadratic Attention** (`TransformerModel`): Uses GPT2 backbone with standard quadratic attention
- **Linear Attention** (`LinearAttentionICLModel`): Implements causal linear attention with ReLU² feature maps

## Data

The data generation creates synthetic regression tasks where:
- Each task has its own regression coefficients β
- Models learn to predict y = X @ β from in-context examples
- Isotropic data: uniform variance across features
- Anisotropic data: different variances per feature (for robustness testing)

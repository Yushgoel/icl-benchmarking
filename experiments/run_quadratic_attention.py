"""Experiment script for quadratic attention ICL benchmarking."""

import torch
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import TransformerModel
from src.utils import load_data, to_serializable, count_parameters
from src.training import train_multiseed
from src.plotting import plot_multiseed_history, final_robustness_check


def run_experiment(config):
    """Run a single experiment configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print model info
    temp_model = TransformerModel(**config['model_params'])
    num_params = count_parameters(temp_model)
    print(f"\n{'='*40}")
    print(f"Model Architecture: {config['model_params']}")
    print(f"Total Parameters:   {num_params:,} ({num_params/1e6:.2f} M)")
    print(f"{'='*40}\n")
    del temp_model

    # Create results directory
    os.makedirs(config['results_dir'], exist_ok=True)
    print(f"Running Experiment on {device}")

    # Load Data
    train_xs, train_ys, test_xs, test_ys = load_data(config['data_path'])

    # Multi-Seed Training
    histories, models = train_multiseed(
        seeds=config['seeds'],
        model_class=TransformerModel,
        train_xs=train_xs,
        train_ys=train_ys,
        test_xs=test_xs,
        test_ys=test_ys,
        config=config,
        device=device
    )

    # Save History
    history_path = os.path.join(config['results_dir'], 'multiseed_history.json')
    with open(history_path, 'w') as f:
        json.dump(histories, f, indent=2, default=to_serializable)
    print(f"Saved histories to {history_path}")

    # Plot Training Stability
    plot_multiseed_history(
        config['model_params']['n_layer'],
        histories,
        model_type="Quadratic",
        save_dir=config['results_dir']
    )

    # Standard vs Anisotropic Robustness Check
    final_robustness_check(
        models,
        test_xs,
        test_ys,
        config['aniso_path'],
        device
    )


if __name__ == '__main__':
    # Example: 6-layer experiment
    config = {
        'seeds': [42, 100, 7, 10, 2025],
        'lr': 3e-4,
        'steps': 10000,
        'batch_size': 64,
        'data_path': 'data/isotropic_data.npz',
        'aniso_path': 'data/anisotropic_data.npz',
        'results_dir': 'results/quadratic_6layer',
        'model_params': {
            'n_dims': 5,
            'n_positions': 10,
            'n_embd': 256,
            'n_layer': 6,
            'n_head': 4,
        }
    }
    
    run_experiment(config)

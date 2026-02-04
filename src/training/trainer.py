import torch
import numpy as np
from tqdm import tqdm
from .evaluator import eval_model


def train_step(model, xs, ys, optimizer, device):
    """
    Training step with proper in-context learning setup.

    Key insight: We predict y_i using context [x0,y0, x1,y1, ..., x_{i-1},y_{i-1}, x_i]
    The causal mask ensures we can't see y_i when predicting it.
    
    Args:
        model: Model to train
        xs: Input features tensor (B, P, D)
        ys: Target values tensor (B, P)
        optimizer: Optimizer
        device: Device to run training on
        
    Returns:
        Loss value (float)
    """
    model.train()
    xs, ys = xs.to(device), ys.to(device)
    optimizer.zero_grad()

    preds = model(xs, ys)
    d = xs.shape[2]

    loss = ((preds - ys) ** 2).mean() / d

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def train_model(model, train_xs, train_ys, test_xs, test_ys, config, device):
    """
    Train a model with evaluation at regular intervals.
    
    Args:
        model: Model to train
        train_xs: Training input features
        train_ys: Training target values
        test_xs: Test input features
        test_ys: Test target values
        config: Configuration dict with keys: lr, steps, batch_size
        device: Device to run training on
        
    Returns:
        Dictionary with training history: {'step': [], 'train_loss': [], 'test_loss': []}
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    history = {'step': [], 'train_loss': [], 'test_loss': []}

    n_train = len(train_xs)
    step = 0
    pbar = tqdm(total=config['steps'], leave=False)

    while step < config['steps']:
        # Random Batch
        idx = torch.randint(0, n_train, (config['batch_size'],))
        loss = train_step(model, train_xs[idx], train_ys[idx], optimizer, device)

        step += 1
        pbar.update(1)

        if step % 500 == 0 or step == config['steps']:
            # Evaluate
            test_mean, _ = eval_model(model, test_xs, test_ys, device)
            # Quick train check
            train_mean, _ = eval_model(model, train_xs[:2000], train_ys[:2000], device)

            history['step'].append(step)
            history['train_loss'].append(train_mean)
            history['test_loss'].append(test_mean)

            pbar.set_postfix({'val_mse': f'{test_mean:.4f}'})

    pbar.close()
    return history


def train_multiseed(seeds, model_class, train_xs, train_ys, test_xs, test_ys, config, device):
    """
    Train multiple models with different random seeds.
    
    Args:
        seeds: List of random seeds
        model_class: Model class to instantiate
        train_xs: Training input features
        train_ys: Training target values
        test_xs: Test input features
        test_ys: Test target values
        config: Configuration dict (must include 'model_params' key)
        device: Device to run training on
        
    Returns:
        Tuple of (all_histories, models) - list of histories and trained models
    """
    all_histories = []
    models = []

    print(f"\nStarting Multi-Seed Training ({len(seeds)} seeds)...")

    for i, seed in enumerate(seeds):
        print(f"--- Run {i+1}/{len(seeds)} | Seed: {seed} ---")

        # 1. Determinism
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 2. Fresh Model
        model = model_class(**config['model_params']).to(device)

        # 3. Train
        hist = train_model(model, train_xs, train_ys, test_xs, test_ys, config, device)

        all_histories.append(hist)
        models.append(model)

    return all_histories, models

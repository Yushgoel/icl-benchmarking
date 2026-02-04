import numpy as np
import torch


def eval_model(model, xs, ys, device, batch_size=64):
    """
    Evaluate model with proper in-context learning - only evaluate last position.
    
    Args:
        model: Model to evaluate
        xs: Input features tensor (N, P, D)
        ys: Target values tensor (N, P)
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        Tuple of (mean_error, std_error) - MSE normalized by dimension
    """
    is_training = model.training
    model.eval()
    all_errors = []
    d = xs.shape[2]

    try:
        with torch.no_grad():
            for i in range(0, len(xs), batch_size):
                batch_xs = xs[i:i+batch_size].to(device)
                batch_ys = ys[i:i+batch_size].to(device)

                preds = model(batch_xs, batch_ys)

                # ICL metric: Only look at the final prediction (n)
                last_pred = preds[:, -1]
                last_target = batch_ys[:, -1]

                # Squared error per sample
                errors = (last_pred - last_target).pow(2)
                errors = errors / d
                all_errors.append(errors.cpu().numpy())
    finally:
        model.train(is_training)

    all_errors = np.concatenate(all_errors)
    return np.mean(all_errors), np.std(all_errors)

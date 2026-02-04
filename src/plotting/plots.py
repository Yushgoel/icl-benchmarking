import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def plot_multiseed_history(n_layers, all_histories, model_type="Transformer", save_dir="results"):
    """
    Plot training curves with confidence intervals across multiple seeds.
    
    Args:
        n_layers: Number of layers (for title)
        all_histories: List of training history dictionaries
        model_type: Type of model ("Transformer" or "Linear")
        save_dir: Directory to save plot
    """
    steps = np.array(all_histories[0]['step'])

    test_losses = np.stack([h['test_loss'] for h in all_histories])
    train_losses = np.stack([h['train_loss'] for h in all_histories])

    mean_test = np.mean(test_losses, axis=0)
    std_test = np.std(test_losses, axis=0)

    mean_train = np.mean(train_losses, axis=0)
    std_train = np.std(train_losses, axis=0)

    n_seeds = len(all_histories)

    se_test = std_test / np.sqrt(n_seeds)
    se_train = std_train / np.sqrt(n_seeds)

    ci_test = 1.96 * se_test
    ci_train = 1.96 * se_train

    plt.figure(figsize=(10, 6))

    plt.plot(steps, mean_train, label="Train Loss (Mean)", linewidth=2)
    plt.fill_between(
        steps,
        mean_train - ci_train,
        mean_train + ci_train,
        alpha=0.15,
        label="Train 95% CI"
    )

    plt.plot(steps, mean_test, label="Test Loss (Mean)", linewidth=2)
    plt.fill_between(
        steps,
        mean_test - ci_test,
        mean_test + ci_test,
        alpha=0.15,
        label="Test 95% CI"
    )

    plt.yscale('log')
    plt.xlabel("Steps")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.title(f"{model_type} {n_layers} layers: train/test loss")
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "multiseed_training_comparison.png"))
    plt.show()


def final_robustness_check(models, test_xs, test_ys, aniso_path, device):
    """
    Compare Average Standard Performance vs Average Anisotropic Performance
    across all trained seeds.
    
    Args:
        models: List of trained models
        test_xs: Standard test input features
        test_ys: Standard test target values
        aniso_path: Path to anisotropic data npz file
        device: Device to run evaluation on
    """
    print(f"\n{'='*40}\nFINAL ROBUSTNESS CHECK\n{'='*40}")

    # Load Anisotropic Data
    if not os.path.exists(aniso_path):
        print(f"Warning: {aniso_path} not found. Skipping anisotropic check.")
        return

    import numpy as np
    import torch
    from ..training.evaluator import eval_model
    
    adata = np.load(aniso_path)
    axs = torch.from_numpy(adata['X']).float()
    ays = torch.from_numpy(adata['y']).float()

    std_means = []
    ani_means = []
    
    for i, m in enumerate(models):
        s_mean, _ = eval_model(m, test_xs, test_ys, device)
        a_mean, _ = eval_model(m, axs, ays, device)
        std_means.append(s_mean)
        ani_means.append(a_mean)
        print(f"Seed {i}: Standard={s_mean:.4f} | Anisotropic={a_mean:.4f}")

    # Aggregate
    avg_std = np.mean(std_means)
    err_std = np.std(std_means)

    avg_ani = np.mean(ani_means)
    err_ani = np.std(ani_means)

    print(f"\nFinal Results (over {len(models)} seeds):")
    print(f"Standard:    {avg_std:.5f} ± {err_std:.5f}")
    print(f"Anisotropic: {avg_ani:.5f} ± {err_ani:.5f}")

    # Bar Plot
    plt.figure(figsize=(6, 5))
    plt.bar(['Standard', 'Anisotropic'], [avg_std, avg_ani],
            yerr=[err_std, err_ani], capsize=10,
            color=['skyblue', 'salmon'], alpha=0.8)
    plt.ylabel("MSE Loss")
    plt.title(f"Robustness (Avg of {len(models)} Seeds)")
    plt.show()

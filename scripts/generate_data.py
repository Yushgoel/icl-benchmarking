import numpy as np

np.random.seed(42)


def generate_data(n_models, n_samples, n_features, x_vars=1, beta_vars=1):
    """
    Generate synthetic ICL data.
    
    Args:
        n_models: Number of different models/tasks
        n_samples: Number of samples per model
        n_features: Number of features (dimensions)
        x_vars: Variance of input features (can be scalar or array)
        beta_vars: Variance of regression coefficients (can be scalar or array)
        
    Returns:
        Tuple of (X, y, beta) where:
        - X: (n_models, n_samples, n_features) input features
        - y: (n_models, n_samples) target values
        - beta: (n_models, n_features) regression coefficients
    """
    if isinstance(beta_vars, (list, np.ndarray)):
        assert len(beta_vars) == n_features
    if isinstance(x_vars, (list, np.ndarray)):
        assert len(x_vars) == n_features
    X = np.random.randn(n_models, n_samples, n_features) * np.sqrt(x_vars)

    beta = np.random.randn(n_models, n_features) * np.sqrt(beta_vars)
    y = np.einsum('ijk,ik->ij', X, beta)
    return X, y, beta


def generate_isotropic_data():
    """Generate isotropic data for training/testing."""
    n_features = 5
    X, y, beta = generate_data(
        n_models=20_000,
        n_samples=10,
        n_features=n_features,
        x_vars=np.array([1] * n_features),
        beta_vars=1
    )
    X_train = X[:16_000]
    y_train = y[:16_000]
    beta_train = beta[:16_000]
    X_test = X[16_000:]
    y_test = y[16_000:]
    beta_test = beta[16_000:]

    np.savez(
        'data/isotropic_data.npz',
        X_train=X_train,
        y_train=y_train,
        beta_train=beta_train,
        X_test=X_test,
        y_test=y_test,
        beta_test=beta_test
    )
    print(f"Generated isotropic data: train={X_train.shape}, test={X_test.shape}")


def generate_anisotropic_data():
    """Generate anisotropic data for robustness testing."""
    n_features = 5
    beta_vars = np.array([0.5, 1, 1.5, 1, 1.75])
    X, y, beta = generate_data(
        n_models=1_000,
        n_samples=10,
        n_features=n_features,
        x_vars=1,
        beta_vars=beta_vars
    )

    np.savez('data/anisotropic_data.npz', X=X, y=y, beta=beta)
    print(f"Generated anisotropic data: {X.shape}")


if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    
    print('Generating isotropic data...')
    generate_isotropic_data()
    
    print('Generating anisotropic data...')
    generate_anisotropic_data()
    
    print('Data generation complete!')

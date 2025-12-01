import numpy as np

np.random.seed(42)

def generate_data(n_models, n_samples, n_features, epsilon_var, x_vars=1, beta_vars=1):
    if isinstance(beta_vars, (list, np.ndarray)):
        assert len(beta_vars) == n_features
    if isinstance(x_vars, (list, np.ndarray)):
        assert len(x_vars) == n_features
    X = np.random.randn(n_models, n_samples, n_features) * np.sqrt(x_vars)
    # epsilon = np.random.randn(n_models, n_samples) * np.sqrt(epsilon_var)

    beta = np.random.randn(n_models, n_features) * np.sqrt(beta_vars)
    y = np.einsum('ijk,ik->ij', X, beta) # + epsilon
    return X, y, beta

def generate_isotropic_data():
    n_features = 5
    X, y, beta = generate_data(n_models=20_000, n_samples=10, n_features=n_features, epsilon_var=0.1, x_vars=np.array([1] * n_features), beta_vars=1)
    X_train = X[:16_000]
    y_train = y[:16_000]
    beta_train = beta[:16_000]
    X_test = X[16_000:]
    y_test = y[16_000:]
    beta_test = beta[16_000:]

    np.savez('data/isotropic_data.npz', X_train=X_train, y_train=y_train, beta_train=beta_train, X_test=X_test, y_test=y_test, beta_test=beta_test)

    
def generate_anisotropic_data():
    n_features = 5
    beta_vars = np.array([0.5, 1, 1.5, 1, 1.75])
    X, y, beta = generate_data(n_models=1_000, n_samples=10, n_features=n_features, epsilon_var=0.1, x_vars=1, beta_vars=beta_vars)

    np.savez('data/anisotropic_data.npz', X=X, y=y, beta=beta)

if __name__ == '__main__':
    # print('Generating isotropic data...')
    # generate_isotropic_data()
    print('Generating anisotropic data...')
    generate_anisotropic_data()
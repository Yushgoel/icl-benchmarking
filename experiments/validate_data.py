"""Script to validate generated data."""

import numpy as np


def validate_isotropic_data():

    isotropic = np.load('data/isotropic_data.npz')
    
    print(f"Keys: {list(isotropic.keys())}")
    print(f"X_train shape: {isotropic['X_train'].shape}")
    print(f"y_train shape: {isotropic['y_train'].shape}")
    print(f"beta_train shape: {isotropic['beta_train'].shape}")
    print(f"X_test shape: {isotropic['X_test'].shape}")
    print(f"y_test shape: {isotropic['y_test'].shape}")
    print(f"beta_test shape: {isotropic['beta_test'].shape}")
    
    # Check that y = X @ beta (within numerical precision)
    y_hat = np.einsum('ijk,ik->ij', isotropic['X_train'], isotropic['beta_train'])
    residuals = y_hat - isotropic['y_train']
    mse = np.sum(residuals**2) / (residuals.shape[0] * residuals.shape[1])
    print(f"\nTrain MSE (should be ~0): {mse:.2e}")
    
    assert mse < 1e-10, "Data validation failed: y != X @ beta"
    print("✓ Isotropic data validation passed!")


def validate_anisotropic_data():
    anisotropic = np.load('data/anisotropic_data.npz')
    
    print(f"Keys: {list(anisotropic.keys())}")
    print(f"X shape: {anisotropic['X'].shape}")
    print(f"y shape: {anisotropic['y'].shape}")
    print(f"beta shape: {anisotropic['beta'].shape}")
    
    # Check beta variances
    beta_vars = np.var(anisotropic['beta'], axis=0)
    print(f"\nBeta variances: {beta_vars}")
    print("Expected: [0.5, 1, 1.5, 1, 1.75]")
    
    y_hat = np.einsum('ijk,ik->ij', anisotropic['X'], anisotropic['beta'])
    residuals = y_hat - anisotropic['y']
    mse = np.sum(residuals**2) / (residuals.shape[0] * residuals.shape[1])
    print(f"\nTrain MSE (should be ~0): {mse:.2e}")
    
    assert mse < 1e-10, "Data validation failed: y != X @ beta"
    print("✓ Anisotropic data validation passed!")



if __name__ == '__main__':
    validate_isotropic_data()
    validate_anisotropic_data()
    print("\n✓ All data validation checks passed!")

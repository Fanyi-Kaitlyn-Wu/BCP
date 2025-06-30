import numpy as np
import pandas as pd
import scipy as sp
import time
from tqdm import tqdm

from sklearn.datasets import load_diabetes, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Sparse Regression Dataset Loader
# -----------------------------------------------------------------------------
def load_traintest_sparsereg(train_frac: float, dataset: str, seed: int):
    """
    Load training and testing data for a sparse regression problem.
    
    Parameters:
    - train_frac: Fraction of the dataset to be used for training.
    - dataset: Either "diabetes" or "boston". For "boston", California housing is used.
    - seed: Random seed for reproducibility.
    
    Returns:
    - x_train, y_train, x_test, y_test, y_plot, n (total number of samples), d (feature dimensionality)
    """
    # Load dataset
    if dataset == "diabetes":
        x, y = load_diabetes(return_X_y=True)
    elif dataset == "boston":
        # Boston dataset was deprecated. Use California housing as alternative.
        data = fetch_california_housing()
        x, y = data.data, data.target
    else:
        raise ValueError("Invalid dataset. Choose 'diabetes' or 'boston'")
    
    n, d = x.shape

    # Standardize features and target for consistency
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    y = (y - np.mean(y)) / np.std(y)

    # Train-test split
    # Multiply train_frac by n and cast to int for train sample count.
    ind_train, ind_test = train_test_split(np.arange(n), train_size=int(train_frac * n), random_state=seed)
    x_train = x[ind_train]
    y_train = y[ind_train]
    x_test = x[ind_test]
    y_test = y[ind_test]

    # Create a plotting grid for y values
    y_plot = np.linspace(np.min(y_train) - 2, np.max(y_train) + 2, 100)

    return x_train, y_train, x_test, y_test, y_plot, n, d


# -----------------------------------------------------------------------------
# Sparse Classification Dataset Loader
# -----------------------------------------------------------------------------
def load_traintest_sparseclass(train_frac: float, dataset: str, seed: int):
    """
    Load training and testing data for a sparse classification problem.
    
    Parameters:
    - train_frac: Fraction of the dataset to be used for training.
    - dataset: Either "breast" or "parkinsons".
    - seed: Random seed for reproducibility.
    
    Returns:
    - x_train, y_train, x_test, y_test, y_plot, n (total number of samples), d (feature dimensionality)
    """
    # Load dataset based on the requested type
    if dataset == "breast":
        x, y = load_breast_cancer(return_X_y=True)
    elif dataset == "parkinsons":
        # Ensure the 'data/parkinsons.data' file is in the correct location.
        data = pd.read_csv('data/parkinsons.data')
        # Replace '?' with NaN and drop any missing values.
        data.replace('?', np.nan, inplace=True)
        data.dropna(axis=0, inplace=True)
        y = data['status'].values  # Convert status to numeric if necessary.
        x = data.drop(columns=['name', 'status']).values
    else:
        raise ValueError("Invalid dataset. Choose 'breast' or 'parkinsons'")
    
    n, d = x.shape

    # Standardize features
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    # Train-test split
    ind_train, ind_test = train_test_split(np.arange(n), train_size=int(train_frac * n), random_state=seed)
    x_train = x[ind_train]
    y_train = y[ind_train]
    x_test = x[ind_test]
    y_test = y[ind_test]

    # The plotting grid for classification typically shows the classes
    y_plot = np.array([0, 1])
    
    return x_train, y_train, x_test, y_test, y_plot, n, d


# -----------------------------------------------------------------------------
# Hierarchical Datasets: Synthetic Data Generator
# -----------------------------------------------------------------------------
def gen_data_hier(n: int, p: int, n_test: int, seed: int, K: int, misspec: bool = False):
    """
    Generate hierarchical data for regression experiments.
    
    Parameters:
    - n: Number of training samples per group.
    - p: Number of features.
    - n_test: Number of test samples per group.
    - seed: Random seed for reproducibility.
    - K: Number of groups.
    - misspec: If True, use a non-Gaussian noise model.
    
    Returns:
    - y (training responses), x (training predictors),
      y_test (test responses), x_test (test predictors),
      beta_true (true coefficients per group),
      sigma_true (noise scales per group), y_plot (grid for plotting).
    """
    theta = np.zeros(p)
    
    # Fix a seed for generating group coefficients
    np.random.seed(24)
    beta_true = np.random.randn(K, p) + theta.reshape(1, -1)
    sigma_true = np.random.exponential(scale=1.0, size=K)
    
    # Generate training data (n samples per group)
    np.random.seed(seed)
    x = np.zeros((n * K, p + 1))  # Extra column for group indicator
    y = np.zeros(n * K)
    
    for k in range(K):
        # If misspecification flag is True, multiply noise by sigma; else standard noise.
        if misspec:
            eps = np.random.randn(n) * sigma_true[k]
        else:
            eps = np.random.randn(n)
        # Generate predictors from normal and append the group index
        x[k * n:(k + 1) * n] = np.hstack((np.random.randn(n, p), np.full((n, 1), k)))
        y[k * n:(k + 1) * n] = np.dot(x[k * n:(k + 1) * n, :p], beta_true[k]) + eps
    
    # Generate test data (n_test samples per group)
    x_test = np.zeros((n_test * K, p + 1))
    y_test = np.zeros(n_test * K)
    
    for k in range(K):
        if misspec:
            eps_test = np.random.randn(n_test) * sigma_true[k]
        else:
            eps_test = np.random.randn(n_test)
        x_test[k * n_test:(k + 1) * n_test] = np.hstack((np.random.randn(n_test, p), np.full((n_test, 1), k)))
        y_test[k * n_test:(k + 1) * n_test] = np.dot(x_test[k * n_test:(k + 1) * n_test, :p], beta_true[k]) + eps_test

    y_plot = np.linspace(-10, 10, 100)
    
    return y, x, y_test, x_test, beta_true, sigma_true, y_plot


# -----------------------------------------------------------------------------
# Hierarchical Dataset Loader: Radon Data (Minnesota)
# -----------------------------------------------------------------------------
def load_traintest_hier(train_frac: float, dataset: str, seed: int):
    """
    Load training and testing data for a hierarchical dataset.
    
    Parameters:
    - train_frac: Fraction of the dataset to be used for training (set 1.0 to use all data for training).
    - dataset: Currently supports "radon".
    - seed: Random seed for reproducibility.
    
    Returns:
    - x_train, y_train, x_test, y_test, y_plot, n (total number of samples), d (feature dimensionality)
    """
    if dataset == "radon":
        # Ensure the data files exist at the specified paths.
        srrs2 = pd.read_csv("data/srrs2.dat", delim_whitespace=True)
        srrs2.columns = srrs2.columns.str.strip()
        srrs_mn = srrs2[srrs2.state == "MN"].copy()

        # Construct fips code from state and county
        srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
        cty = pd.read_csv("data/cty.dat", delim_whitespace=True)
        cty_mn = cty[cty.st == "MN"].copy()
        cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips

        srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
        srrs_mn.drop_duplicates(subset="idnum", inplace=True)
        # Unique log-transformed uranium values (if needed later)
        _ = np.log(srrs_mn.Uppm).unique()

        n = len(srrs_mn)
        # Clean county names
        srrs_mn.county = srrs_mn.county.str.strip()
        mn_counties = srrs_mn.county.unique()
        counties = len(mn_counties)
        county_lookup = {county: idx for idx, county in enumerate(mn_counties)}
        # Map county codes
        srrs_mn["county_code"] = srrs_mn.county.replace(county_lookup).values

        radon = srrs_mn.activity  # Original radon activity
        log_radon = np.log(radon + 0.1).values  # Log-transformed values
        floor = srrs_mn.floor.values

        # Preprocess predictors: floor and county code
        x = np.column_stack((floor, srrs_mn["county_code"].values.astype(int)))
        y = log_radon
    else:
        raise ValueError("Invalid dataset. Only 'radon' is supported.")

    n, d = x.shape

    # Train-test split; stratify using county codes for balanced representation if possible.
    if train_frac == 1.0:
        ind_train = np.arange(n)
        ind_test = np.array([], dtype=int)
    else:
        ind_train, ind_test = train_test_split(np.arange(n),
                                               train_size=int(train_frac * n),
                                               random_state=seed,
                                               stratify=x[:, 1])
    x_train = x[ind_train]
    y_train = y[ind_train]
    x_test = x[ind_test]
    y_test = y[ind_test]

    y_plot = np.linspace(-6, 6, 100)
    
    return x_train, y_train, x_test, y_test, y_plot, n, d

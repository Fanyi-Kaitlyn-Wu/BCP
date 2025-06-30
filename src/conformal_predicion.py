import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV

# Helper function for residuals
def absolute_residuals(y_true, y_pred):
    return np.abs(y_true - y_pred)

# Split conformal with proper CV for lambda selection
def split_conformal_prediction(X_train, y_train, X_cal, y_cal, alpha=0.2):
    """Split conformal prediction with CV for lambda selection"""
    # Use half the training data for CV, as in the paper
    X_train_half, _, y_train_half, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
    
    # Find optimal lambda using cross-validation
    lasso_cv = LassoCV(cv=5, random_state=42)
    lasso_cv.fit(X_train_half, y_train_half)
    best_alpha = lasso_cv.alpha_
    
    # Use fixed alpha from paper if needed
    # best_alpha = 0.004
    
    # Fit model on training data
    model = Lasso(alpha=best_alpha, random_state=42)
    model.fit(X_train, y_train)
    
    # Compute residuals on calibration set
    y_cal_pred = model.predict(X_cal)
    residuals_cal = absolute_residuals(y_cal, y_cal_pred)
    
    # Quantile of residuals
    q_value = np.quantile(residuals_cal, 1 - alpha)
    
    return model, q_value, best_alpha

def predict_with_split_cp(model, q_value, X_test):
    """Generate prediction intervals using split conformal"""
    y_pred = model.predict(X_test)
    lower = y_pred - q_value
    upper = y_pred + q_value
    return lower, upper, y_pred

def full_conformal_prediction(X, y, X_test, alpha=0.2):
    """Full conformal prediction method with fixed alpha from paper"""
    best_alpha = 0.004  # Fixed value from paper
    
    model = Lasso(alpha=best_alpha, random_state=42)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    residuals = absolute_residuals(y, y_pred)
    threshold = np.quantile(residuals, 1 - alpha)
    
    test_preds = model.predict(X_test)
    lower = test_preds - threshold
    upper = test_preds + threshold
    
    return lower, upper, test_preds, best_alpha

# Grid conformal prediction for a single test point
def grid_conformal_prediction(X, y, x_test, alpha=0.2, best_alpha=0.004, grid_size=100):
    """
    Implementation of full conformal prediction using the grid method
    for a single test point as described in the paper.
    """
    # 1) Fit model on full data
    model = Lasso(alpha=best_alpha, random_state=42, max_iter=10000)
    model.fit(X, y)
    
    # 2) Get point prediction
    y_test_pred = model.predict(x_test.reshape(1, -1))[0]
    
    # 3) Define grid around predicted value as in paper
    y_min = np.min(y)
    y_max = np.max(y)
    grid_points = np.linspace(y_min - 2, y_max + 2, grid_size)
    
    # 4) For each potential y value, check if it would be conformal
    conforming_points = []
    
    for y_grid in grid_points:
        # Augment data with test point and potential y value
        X_aug = np.vstack([X, x_test])
        y_aug = np.append(y, y_grid)
        
        # Fit model on augmented data
        aug_model = Lasso(alpha=best_alpha, random_state=42, max_iter=10000)
        aug_model.fit(X_aug, y_aug)
        
        # Compute residuals
        y_aug_pred = aug_model.predict(X_aug)
        residuals = absolute_residuals(y_aug, y_aug_pred)
        
        # Compute nonconformity score for test point
        test_score = residuals[-1]
        
        # Compute rank of test point among all residuals
        rank = (residuals <= test_score).mean()
        
        # If rank is high enough, this y value is in the prediction interval
        if rank >= alpha:
            conforming_points.append(y_grid)
    
    # 5) Determine interval bounds
    if len(conforming_points) > 0:
        lower = min(conforming_points)
        upper = max(conforming_points)
    else:
        # Fallback if no points are conforming
        lower = y_test_pred - 2
        upper = y_test_pred + 2
    
    return lower, upper, y_test_pred
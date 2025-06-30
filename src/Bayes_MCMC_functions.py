import jax.numpy as jnp
from jax import jit
from functools import partial 
import numpy as np
import scipy as sp
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp

## BAYESIAN CREDIBLE INTERVALS FROM MCMC SAMPLES ##
# Compute Bayesian central 1-alpha credible interval.
@jit
def compute_bayes_band_MCMC(alpha, y_plot, cdf_pred):
    """
    Compute the Bayesian central credible interval from MCMC samples.

    Parameters:
    - alpha: float, significance level (e.g., 0.1 for a 90% credible interval)
    - y_plot: array-like, grid of values where the predictive CDF is evaluated
    - cdf_pred: array-like, predictive CDF samples with shape (num_samples, len(y_plot))

    Returns:
    - band_bayes: a 2-element array with the lower and upper bounds of the credible interval.
    """
    # Average the predictive CDF across MCMC samples.
    cdf_pred = jnp.mean(cdf_pred, axis=0)

    # Find the y_plot values corresponding to the lower and upper quantiles.
    left = y_plot[jnp.argmin(jnp.abs(cdf_pred - alpha / 2))]
    right = y_plot[jnp.argmin(jnp.abs(cdf_pred - (1 - alpha / 2)))]
    band_bayes = jnp.array([left, right])
    
    return band_bayes

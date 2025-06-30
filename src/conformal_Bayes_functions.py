import jax.numpy as jnp
from jax import jit
from jax.scipy.special import logsumexp
import numpy as np  # kept for non-JAX functions if needed
import scipy as sp
from jax.scipy.stats import norm
from functools import partial

## CONFORMAL FROM MCMC SAMPLES ##
### JAX IMPLEMENTATION

@jit
def compute_rank_IS(logp_samp_n, logwjk):
    """
    Compute unnormalized conformal ranks based on importance sampling.
    
    Parameters:
    - logp_samp_n: array of shape (B, n) with log probability samples.
    - logwjk: array of shape (n_plot, B) with log weights.
    
    Returns:
    - rank_cp: array of shape (n_plot,) with computed ranks.
    """
    # Number of samples and grid points.
    n = jnp.shape(logp_samp_n)[1]  # logp_samp_n is (B, n)
    n_plot = jnp.shape(logwjk)[0]
    
    # Compute importance sampling weights and normalization.
    wjk = jnp.exp(logwjk)
    Zjk = jnp.sum(wjk, axis=1, keepdims=True)
    
    # Compute predictive non-conformal scores for the n training points.
    p_cp = jnp.dot(wjk / Zjk, jnp.exp(logp_samp_n))  # shape: (n_plot, n)
    # Compute predictive score for the new observation.
    p_new = jnp.sum(wjk**2, axis=1, keepdims=True) / Zjk  # shape: (n_plot, 1)
    
    # Combine predictions and compute rank by comparing each grid point’s value
    # with the new observation's predictive score.
    pred_tot = jnp.concatenate((p_cp, p_new), axis=1)
    rank_cp = jnp.sum(pred_tot <= pred_tot[:, -1][:, None], axis=1)
    
    return rank_cp

@jit
def compute_cb_region_IS(alpha, logp_samp_n, logwjk):
    """
    Compute the conformal prediction region based on a significance level alpha.
    
    Parameters:
    - alpha: significance level (e.g., 0.1 for 90% confidence).
    - logp_samp_n: array of shape (B, n) with log probability samples.
    - logwjk: array of shape (n_plot, B) with log weights.
    
    Returns:
    - region_true: boolean array of shape (n_plot,) indicating if each grid point
      is in the conformal prediction region.
    """
    n = jnp.shape(logp_samp_n)[1]
    rank_cp = compute_rank_IS(logp_samp_n, logwjk)
    region_true = rank_cp > alpha * (n + 1)
    return region_true

@jit
def diagnose_IS_weights(logp_samp_n, logwjk):
    """
    Diagnose importance sampling weights by computing the Effective Sample Size (ESS)
    and variance of the predictive estimates.
    
    Parameters:
    - logp_samp_n: array of shape (B, n) with log probability samples.
    - logwjk: array of shape (n_plot, B) with log weights.
    
    Returns:
    - ESS: array of ESS values for each grid point.
    - var: array of variances for each grid point.
    """
    n = jnp.shape(logp_samp_n)[1]
    n_plot = jnp.shape(logwjk)[0]
    
    # Reshape log weights for the computation.
    logwjk = logwjk.reshape(n_plot, -1, 1)  # new shape: (n_plot, B, 1)
    logZjk = logsumexp(logwjk, axis=1)  # shape: (n_plot, 1)
    
    # Compute the log predictive for new observations.
    logp_new = logsumexp(2 * logwjk, axis=1) - logZjk  # shape: (n_plot, 1)
    
    # Compute normalized weights.
    wjk = jnp.exp(logwjk - logZjk[:, None, None])
    ESS = 1.0 / jnp.sum(wjk**2, axis=1)
    
    # Compute the variance of the predictive estimates.
    var = jnp.sum(wjk**2 * (wjk - jnp.exp(logp_new)[:, None, None])**2, axis=1)
    
    return ESS, var

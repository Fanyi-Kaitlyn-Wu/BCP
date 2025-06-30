from jax.scipy.special import logsumexp
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

# ------------------------------------------------------------------
# 1.  Build    logp_samp_n   ——  training-point log-likelihoods
#               shape (B, n)
# ------------------------------------------------------------------
@jit
def build_logp_samp_n(X_train,             # (n, d)
                      y_train,             # (n,)
                      theta_samples,       # (B, d)
                      intercept_samples,   # (B,)
                      tau_samples):        # (B,)
    """
    log f_{θ(t)}(Y_i | X_i)  for every posterior draw t and every data point i.
    """
    means = jnp.matmul(theta_samples, X_train.T) + intercept_samples[:, None]  # (B, n)
    return norm.logpdf(y_train, means, tau_samples[:, None])                   # (B, n)

# ------------------------------------------------------------
# 0.  reference computation: plain log-likelihoods
# ------------------------------------------------------------
def build_logpdf_matrix(X_train, y_train, theta_samples, intercept_samples, tau_samples):
    means = jnp.matmul(theta_samples, X_train.T) + intercept_samples[:, None]          # (B, n)
    return norm.logpdf(y_train, means, tau_samples[:, None])          # (B, n)

# ------------------------------------------------------------
# 1.  Tempered log-likelihood   lt,λ = λ lt      (default λ = 0.7)
# ------------------------------------------------------------
@jit
def build_logp_samp_n_tempered(X_train, y_train, theta_samples, intercept_samples, tau_samples, lam=0.7):
    return lam * build_logpdf_matrix(X_train, y_train, theta_samples, intercept_samples, tau_samples)

# ------------------------------------------------------------
# 2.  Studentised log-likelihood (common σ = posterior mean)
# ------------------------------------------------------------
@jit
def build_logp_samp_n_studentised(X_train, y_train, theta_samples, intercept_samples, tau_samples):
    tau_bar = jnp.mean(tau_samples)                               # scalar
    means = jnp.matmul(theta_samples, X_train.T) + intercept_samples[:, None]
    return norm.logpdf(y_train, means, tau_bar)               # (B, n)  broadcast

# ------------------------------------------------------------
# 3.  PSIS-LOO log predictive density  (constant across draws)
#      l_loo_i = log ∑t wit ft(Yi|Xi)  with  wit ∝ 1/ft(Yi|Xi)
# ------------------------------------------------------------
@jit
def build_logp_samp_n_loo(X_train, y_train, theta_samples, intercept_samples, tau_samples):
    l = build_logpdf_matrix(X_train, y_train, theta_samples, intercept_samples, tau_samples)  # (B, n)
    # importance weights  wit ∝ exp(-lit)
    logw = -l                                         # (B, n)
    logw -= logsumexp(logw, axis=0, keepdims=True)    # normalise over t
    # PSIS-LOO estimate for each i: log Σt wit ft = log Σt exp(logw+l)
    loo_logp = logsumexp(logw + l, axis=0)            # (n,)
    # replicate to shape (B, n) so downstream code's shape conventions hold
    return jnp.broadcast_to(loo_logp, l.shape)

# ------------------------------------------------------------
# 4.  Two-sided tail log-score  l = log 2·min{F, 1−F}
# ------------------------------------------------------------
@jit
def build_logp_samp_n_two_sided(X_train, y_train, theta_samples, intercept_samples, tau_samples, eps=1e-12):
    means = jnp.matmul(theta_samples, X_train.T) + intercept_samples[:, None]
    cdf   = norm.cdf(y_train, means, tau_samples[:, None])            # (B, n)
    tail  = 2.0 * jnp.minimum(cdf, 1.0 - cdf) + eps   # add eps to avoid log0
    return jnp.log(tail)

# ------------------------------------------------------------
# 5.  Variance-penalised log-score  lt,β = lt − β log τt
# ------------------------------------------------------------
@jit
def build_logp_samp_n_var_penalised(X_train, y_train, theta_samples, intercept_samples, tau_samples, beta=0.5):
    base = build_logpdf_matrix(X_train, y_train, theta_samples, intercept_samples, tau_samples)
    penalty = beta * jnp.log(tau_samples)[:, None]              # (B, 1) → broadcast
    return base - penalty

@jit
def build_logp_samp_n_postpred(X_train, y_train,
                               theta_samples,      # (B, d)
                               intercept_samples,  # (B,)
                               tau_samples):       # (B,)
    """
    Posterior-predictive *non-conformity* matrix
        log 𝑝̂(Y_i|X_i)   (broadcast to shape (B, n))

    The broadcast keeps the (B, n) shape required by
    `compute_rank_IS`, but every column is identical
    because the predictive has already been averaged
    over draws.
    """

    # 1. per-draw log-likelihoods  ℓ_{t,i}
    means = jnp.matmul(theta_samples, X_train.T) + intercept_samples[:, None]  # (B, n)
    log_like = norm.logpdf(y_train, means, tau_samples[:, None])               # (B, n)

    # 2. log posterior-predictive for each i :  log(1/B Σ_t exp ℓ_{t,i})
    log_pp = logsumexp(log_like, axis=0) - jnp.log(log_like.shape[0])          # (n,)

    # 3. broadcast back to (B, n) so the shape matches older code
    return jnp.broadcast_to(log_pp, log_like.shape)

@jit
def build_studentised_residual(X_train, y_train,
                               theta_samples, intercept_samples, tau_samples):
    """
    |y_i - E[μ_i]| / sqrt(E[τ^2])      broadcast to (B, n)
    """
    # posterior means across draws
    mu_bar   = (theta_samples @ X_train.T + intercept_samples[:, None]).mean(0)   # (n,)
    sigma2_bar = jnp.square(tau_samples).mean()                                   # scalar

    score = jnp.abs(y_train - mu_bar) / jnp.sqrt(sigma2_bar)                      # (n,)
    return jnp.broadcast_to(score, (theta_samples.shape[0], y_train.shape[0]))

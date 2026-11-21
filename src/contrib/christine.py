import numpy as np



def _clean_returns(returns: np.ndarray) -> np.ndarray:
    """Validate the returns matrix and remove rows containing NaNs."""

    if not isinstance(returns, np.ndarray) or returns.ndim != 2:
        raise ValueError(
            "returns must be a two-dimensional NumPy array."
        )

    valid_rows = ~np.isnan(returns).any(axis=1)
    clean_returns = returns[valid_rows]

    if clean_returns.shape[0] < 2:
        raise ValueError(
            "At least two observations are required after removing NaN rows."
        )

    if clean_returns.shape[1] == 0:
        raise ValueError("At least one strategy is required.")

    if not np.isfinite(clean_returns).all():
        raise ValueError("Returns must contain only finite values.")

    if np.any(np.all(clean_returns == clean_returns[0], axis=0)):
        raise ValueError(
            "Returns must not contain constant strategy columns."
        )

    return clean_returns

def effective_trials(returns: np.ndarray) -> float:
    """
    Estimate the effective number of independent strategy trials.

    Parameters
    ----------
    returns : np.ndarray
        Two-dimensional array of hourly returns, with observations
        in rows and strategy variants in columns. Rows containing
        NaN values are removed.

    Returns
    -------
    float
        Effective number of trials calculated as the squared sum
        of correlation matrix eigenvalues divided by the sum of
        their squares. A single valid strategy returns 1.0.

    Raises
    ------
    ValueError
        If the input is not two-dimensional, has insufficient
        observations, contains no strategies, contains infinite
        values, or contains constant strategy columns.
    """

    clean_returns = _clean_returns(returns)

    if clean_returns.shape[1] == 1:
        return 1.0

    cor_matrix = np.corrcoef(clean_returns, rowvar=False)

    eig_vals = np.linalg.eigvalsh(cor_matrix)
    numerator = (np.sum(eig_vals))**2
    denominator = np.sum(eig_vals**2)

    return numerator/denominator

def sharpe_spread(returns: np.ndarray) -> float:
    """
    Calculate the sample variance of hourly Sharpe ratios.

    Parameters
    ----------
    returns : np.ndarray
        Two-dimensional array of hourly returns, with observations
        in rows and strategy variants in columns. Rows containing
        NaN values are removed.

    Returns
    -------
    float
        Sample variance of the per-strategy hourly Sharpe ratios.
        Both the return standard deviations and the variance across
        Sharpe ratios are calculated using ddof=1. The Sharpe
        ratios are not annualized.

    Raises
    ------
    ValueError
        If the input is invalid, contains insufficient observations,
        contains non-finite values after NaN removal, contains
        constant strategy columns, or has fewer than two strategies.
    """

    clean_returns = _clean_returns(returns)

    if clean_returns.shape[1] < 2:
        raise ValueError(
            "At least two strategies are required to calculate Sharpe spread."
        )
    
    means = np.mean(clean_returns, axis=0)
    stds = np.std(clean_returns, axis=0, ddof=1)
    sharpes = means/stds

    sample_var = np.var(sharpes, ddof=1)

    return float(sample_var)

def bootstrap_hourly_sharpes(
    returns,
    block_size=24,
    n_bootstrap=2000,
    seed=42
):
    """Estimate hourly Sharpe variance under a zero-mean null."""

    rng = np.random.default_rng(seed)

    centered_returns = returns - np.mean(returns)
    n = len(centered_returns)


    if n < 2:
        raise ValueError("At least two returns are required.")

    if block_size < 1 or block_size > n:
        raise ValueError("block_size must be between 1 and the number of returns.")

    if np.all(returns == returns[0]):
        raise ValueError("Cannot calculate Sharpe ratios for constant returns.")


    blocks_needed = int(np.ceil(n / block_size))
    bootstrap_sharpes = []

    for _ in range(n_bootstrap):

        starting_indices = rng.integers(
            low=0,
            high=n - block_size + 1,
            size=blocks_needed
        )

        blocks = [
            centered_returns[i:i + block_size]
            for i in starting_indices
        ]

        bootstrap_sample = np.concatenate(blocks)[:n]

        sr = np.mean(bootstrap_sample) / np.std(
            bootstrap_sample, ddof=1
        )
        bootstrap_sharpes.append(sr)

    bootstrap_sharpes = np.array(bootstrap_sharpes)

    return bootstrap_sharpes

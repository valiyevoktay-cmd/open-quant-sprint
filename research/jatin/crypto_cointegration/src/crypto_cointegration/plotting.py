import matplotlib.pyplot as plt

from .models import kalman_hedge_ratio


def plot_kalman_beta(log_prices, symbol_y, symbol_x, formation_hours=720,
                     testing_hours=720, delta=1e-5, obs_var=1e-3):
    end = formation_hours + testing_hours
    y = log_prices[symbol_y].iloc[:end].to_numpy()
    x = log_prices[symbol_x].iloc[:end].to_numpy()
    beta = kalman_hedge_ratio(y, x, delta=delta, obs_var=obs_var)
    timestamps = log_prices.index[:end]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(timestamps, beta, linewidth=1, label="Kalman beta")
    ax.axvline(timestamps[formation_hours], color="red", linestyle="--",
               label="Formation / testing boundary")
    ax.set(title=f"{symbol_y}–{symbol_x}: Kalman hedge ratio",
           xlabel="Timestamp (UTC)", ylabel="Beta")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax

"""
Recover the unknown parameters (theta, M, X) of the parametric curve

    x(t) = t*cos(theta) - exp(M*|t|)*sin(0.3*t)*sin(theta) + X
    y(t) = 42 + t*sin(theta) + exp(M*|t|)*sin(0.3*t)*cos(theta),   6 < t < 60

given only a cloud of (x, y) points sampled from the curve (xy_data.csv),
with no accompanying t values.

APPROACH
--------
Write u = t and v = exp(M*|t|)*sin(0.3*t). Then the model is exactly a
2D rotation (by theta) plus a translation (by X, 42) applied to the point
(u, v):

    [x - X ]   [cos(theta)  -sin(theta)] [u]
    [y - 42] = [sin(theta)   cos(theta)] [v]

Rotation matrices are orthogonal, so this is invertible for ANY trial
(theta, X) -- we don't need to search for t separately per point. For a
candidate (theta, X), inverting the rotation gives a direct estimate of
t for every data point simultaneously:

    t_pred   =  cos(theta)*(x-X) + sin(theta)*(y-42)
    v_actual = -sin(theta)*(x-X) + cos(theta)*(y-42)

The parameters are correct exactly when
    v_actual == exp(M*|t_pred|) * sin(0.3*t_pred)
holds for every point. That collapses the problem to an ordinary
3-parameter least-squares fit (theta, M, X) -- no per-point unknowns,
no correspondence problem.

We minimize sum((v_actual - v_pred)^2) with:
  1. A global optimizer (differential evolution) over the given bounds
     to avoid the local minima created by the sin(0.3t) periodicity.
  2. A local Nelder-Mead polish to converge to machine precision.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = '/mnt/user-data/uploads/xy_data.csv'
BOUNDS = [(0, 50), (-0.05, 0.05), (0, 100)]  # theta(deg), M, X


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df['x'].values, df['y'].values


def residuals(params, x, y):
    theta_deg, M, X = params
    th = np.radians(theta_deg)
    xp, yp = x - X, y - 42.0
    t_pred = xp * np.cos(th) + yp * np.sin(th)
    v_actual = -xp * np.sin(th) + yp * np.cos(th)
    v_pred = np.exp(M * np.abs(t_pred)) * np.sin(0.3 * t_pred)
    return v_actual - v_pred


def loss(params, x, y):
    r = residuals(params, x, y)
    return np.sum(r ** 2)


def fit(x, y):
    de = differential_evolution(
        loss, BOUNDS, args=(x, y), tol=1e-15, seed=42,
        maxiter=3000, popsize=60, mutation=(0.3, 1.7),
        recombination=0.9, polish=True, updating='deferred',
    )
    refined = minimize(
        loss, de.x, args=(x, y), method='Nelder-Mead',
        options={'xatol': 1e-12, 'fatol': 1e-16, 'maxiter': 200000, 'maxfev': 200000},
    )
    return refined.x, refined.fun


def curve_xy(t, theta_deg, M, X):
    th = np.radians(theta_deg)
    env = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(th) - env * np.sin(th) + X
    y = 42 + t * np.sin(th) + env * np.cos(th)
    return x, y


def main():
    x, y = load_data()
    print(f"Loaded {len(x)} data points from {DATA_PATH}")

    params, final_loss = fit(x, y)
    theta_deg, M, X = params
    r = residuals(params, x, y)

    print("\n=== Recovered parameters ===")
    print(f"theta = {theta_deg:.6f} deg = {np.radians(theta_deg):.6f} rad")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.6f}")
    print(f"sum-sq loss = {final_loss:.3e}, max|residual| = {np.max(np.abs(r)):.3e}")

    # Round to the evident "clean" values and confirm they reproduce the
    # data equally well (i.e. rounding noise, not model mismatch).
    clean = (round(theta_deg), round(M, 2), round(X))
    r_clean = residuals(clean, x, y)
    print(f"\nClean-value check theta={clean[0]}, M={clean[1]}, X={clean[2]}:")
    print(f"max|residual| = {np.max(np.abs(r_clean)):.3e}  (~data-precision noise)")

    # Forward-simulate x,y from the recovered t per point and compare to
    # the raw data directly -- this mirrors the L1-distance grading metric.
    th = np.radians(clean[0])
    t_rec = (x - clean[2]) * np.cos(th) + (y - 42.0) * np.sin(th)
    x_hat, y_hat = curve_xy(t_rec, *clean)
    l1 = np.mean(np.abs(x - x_hat) + np.abs(y - y_hat))
    print(f"Mean L1 reconstruction error (x+y): {l1:.3e}")
    print(f"Recovered t range: [{t_rec.min():.3f}, {t_rec.max():.3f}]  (spec: 6 < t < 60)")

    # Plot: raw data vs. fitted curve
    t_dense = np.linspace(6, 60, 2000)
    x_fit, y_fit = curve_xy(t_dense, *clean)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=8, color='#333333', alpha=0.5, label='given data (xy_data.csv)')
    ax.plot(x_fit, y_fit, color='#e0722a', linewidth=2,
            label=f'fitted curve  θ={clean[0]}°, M={clean[1]}, X={clean[2]}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Data vs. fitted parametric curve')
    ax.legend()
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    fig.savefig('/home/claude/fit_plot.png', dpi=150)
    print("\nSaved plot to fit_plot.png")

    # Desmos / LaTeX submission string, in the exact format requested
    theta_rad = np.radians(clean[0])
    latex = (
        f"\\left(t*\\cos({theta_rad:.6f})-e^{{{clean[1]}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\sin({theta_rad:.6f})+{clean[2]},"
        f"42+t*\\sin({theta_rad:.6f})+e^{{{clean[1]}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\cos({theta_rad:.6f})\\right)"
    )
    print("\n=== Desmos / LaTeX submission string ===")
    print(latex)

    return clean, latex


if __name__ == '__main__':
    main()

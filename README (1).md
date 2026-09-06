# Parametric Curve Fitting — Solution

## Problem

Recover `theta`, `M`, `X` in

```
x(t) = t*cos(theta) - exp(M*|t|)*sin(0.3t)*sin(theta) + X
y(t) = 42 + t*sin(theta) + exp(M*|t|)*sin(0.3t)*cos(theta),    6 < t < 60
```

given only the 1500 `(x, y)` points in `xy_data.csv` (no `t` values, no
correspondence to the curve given), subject to:

```
0 deg < theta < 50 deg
-0.05 < M < 0.05
0 < X < 100
```

## Approach

**Key observation:** let `u = t` and `v = exp(M|t|)*sin(0.3t)`. Then the
model is exactly a rigid **rotation by `theta`** plus a **translation by
`(X, 42)`** applied to the point `(u, v)`:

```
[x - X ]   [cos(theta)  -sin(theta)] [u]
[y - 42] = [sin(theta)   cos(theta)] [v]
```

Rotation matrices are orthogonal, so this is invertible for *any* trial
`(theta, X)` — there's no need to search for `t` separately per point.
Inverting the rotation gives a direct estimate of `t` for every data
point at once:

```
t_pred   =  cos(theta)*(x-X) + sin(theta)*(y-42)
v_actual = -sin(theta)*(x-X) + cos(theta)*(y-42)
```

The parameters are correct exactly when `v_actual == exp(M*|t_pred|) *
sin(0.3*t_pred)` holds for every point simultaneously. This turns a
messy "unknown correspondence" problem into an ordinary 3-parameter
least-squares fit — no per-point unknowns.

**Optimization** (`solve_parametric_fit.py`):
1. Minimize `sum((v_actual - v_pred)^2)` over `(theta, M, X)` with
   `scipy.optimize.differential_evolution` across the full given bounds
   (a global search, needed because `sin(0.3t)` introduces periodicity
   that a purely local optimizer could get stuck on).
2. Polish the best candidate with a local Nelder-Mead run to converge
   to machine precision.

## Result

| Parameter | Value |
|---|---|
| `theta` | **30°**  (`= pi/6 = 0.523599 rad`) |
| `M` | **0.03** |
| `X` | **55** |

All three sit comfortably inside the required bounds. Plugging these
back in and forward-simulating `x(t), y(t)` reproduces every point in
`xy_data.csv` to within ~4e-5 (i.e. right at the rounding precision of
the CSV itself) — see `fit_plot.png`, where the fitted curve lies
exactly on top of the given data.

**Desmos / LaTeX submission string:**

```
\left(t*\cos(0.523599)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.523599)+55,42+t*\sin(0.523599)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.523599)\right)
```
with domain `6 <= t <= 60`.

## Files

- `solve_parametric_fit.py` — full pipeline: load data, fit, validate, plot.
- `fit_plot.png` — given data vs. fitted curve.

## Running it

```bash
pip install numpy pandas scipy matplotlib
python solve_parametric_fit.py
```

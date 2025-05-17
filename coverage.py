# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# %%
rng = np.random.default_rng(42)

# %%
lh_pos_x, lh_pos_y = 0.0, 5.0

# %%
import sympy as sp

x = sp.symbols("x", real=True)
x0 = sp.symbols("x_0", real=True)
gamma = sp.symbols("gamma", positive=True)

nll = -sp.log(1 / (sp.pi * gamma * (1 + ((x - x0) / gamma)**2))).simplify()
jac_sympy = sp.lambdify((x, x0, gamma), sp.simplify(sp.Matrix([nll]).jacobian([x0, gamma])), cse=True)


# %%
def get_sampling_distribution(n, size, rng):
    bounds = [
        (-100, 100),
        (0.01, 100),
    ]
    
    res = []
    for _ in range(size):
        X = stats.cauchy.rvs(loc=lh_pos_x, scale=lh_pos_y, size=n, random_state=rng)

        # Starting parameters
        loc, scale = np.median(X), max(stats.iqr(X) / 2.0, 1e-2)

        res_opt = opt.minimize(
            lambda x: -stats.cauchy.logpdf(X, x[0], x[1]).sum(),
            x0=(loc, scale),
            bounds=bounds,
            jac=lambda x: jac_sympy(X, x[0], x[1]).sum(axis=-1),
        )

        if not res_opt.success:
            print(res_opt.message)
            res_opt = opt.minimize(
                lambda x: -stats.cauchy.logpdf(X, x[0], x[1]).sum(),
                x0=(loc, scale),
                bounds=bounds,
                method="Nelder-Mead",
                options={"maxiter": 2000},
            )
            assert res_opt.success

        delta = -stats.cauchy.logpdf(X, lh_pos_x, lh_pos_y).sum() - res_opt.fun
        res.append(2 * delta.item())

    return np.array(res)


# %%
N = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20] + [2**x for x in range(5, 11)]
N

# %%
# %%time
res = Parallel(n_jobs=-1)(
    delayed(get_sampling_distribution)(
        n, 100_000, np.random.default_rng(42 + n)
    )
    for n in N
)

# %%
cov_68 = []
cov_95 = []
for n, dist in zip(N, res):
    cov_68.append(np.mean(dist < stats.chi2.ppf(0.68, df=2)))
    cov_95.append(np.mean(dist < stats.chi2.ppf(0.95, df=2)))

# %%
fig, axs = plt.subplots(sharex=True, nrows=2)

axs[0].set_xscale("log")
axs[0].axhline(0.68, c="k", ls="--")
axs[0].plot(N, cov_68)

axs[1].set_xscale("log")
axs[1].axhline(0.95, c="k", ls="--")
axs[1].plot(N, cov_95)

axs[1].set_xlabel("Sample size")
axs[0].set_ylabel("Coverage")
axs[1].set_ylabel("Coverage")

# %%
# 16 jobs -> 14 min (linux native: 14 min) (with analytic jacobian: 6 min)
# 8 jobs -> 20 minutes
# 4 jobs -> 34 min
# Silly for loop -> 1h 37 min

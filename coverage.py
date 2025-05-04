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

# %%
rng = np.random.default_rng(42)

# %%
lh_pos_x, lh_pos_y = 0.0, 5.0


# %%
def get_starting_params(X):
    return np.median(X) + 0.1, stats.iqr(X) / 2.0 + 0.1


# %%
def get_sampling_distribution(n, size):
    res = []
    
    for _ in range(size):
        X = stats.cauchy.rvs(loc=lh_pos_x, scale=lh_pos_y, size=n, random_state=rng)    

        res_opt = opt.minimize(
            lambda x: -2 * stats.cauchy.logpdf(X, x[0], x[1]).sum(),
            x0=get_starting_params(X),
            bounds=[(-100, 100), (0.01, 100)],
        )

        if not res_opt.success:
            print(res_opt.message)
            res_opt = opt.minimize(
                lambda x: -2 * stats.cauchy.logpdf(X, x[0], x[1]).sum(),
                x0=get_starting_params(X),
                bounds=[(-100, 100), (0.01, 100)],
                method="Nelder-Mead",
                options={"maxiter": 2000},
            )
            assert res_opt.success    
            
        delta = -2 * stats.cauchy.logpdf(X, lh_pos_x, lh_pos_y).sum() - res_opt.fun
        res.append(delta.item())

    return np.array(res)


# %%
N = [5, 10, 20, 40, 80, 160, 320]

# %%
# %%time
lam = get_sampling_distribution(1000, 10000)

# %%
x = np.linspace(0, 10)
density = stats.chi2.pdf(x, df=2) / stats.chi2.cdf(10, df=2)

fig, ax = plt.subplots()
ax.hist(lam, bins=40, range=(0, 10), density=True)
ax.plot(x, density)

# %%
np.mean(lam < stats.chi2.ppf(0.68, df=2))

# %%
np.mean(lam < stats.chi2.ppf(0.95, df=2))

# %%

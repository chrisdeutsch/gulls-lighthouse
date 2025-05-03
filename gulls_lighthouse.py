# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% [markdown]
# # Gull's Lighthouse
#
# The lighthouse problem:
#
# > A lighthouse is somewhere off a piece of straight coastline at position $x_0$ along the coast and a distance $y$ out to sea.
# > It emits a series of short, highly collimated flashes at random intervals and hence at random azimuths.
# > These pulses are intercepted on the coast by photo-detectors that record only the fact that a flash has occurred, but *not* the azimuth from which it came.
# > $N$ Flashes have so far been recorded at positions $\{x_i,\ i=1, \dots ,N\}$.
# > Where is the lighthouse?
#
# Stephen F. Gull (1988): Bayesian Inductive Inference and Maximum Entropy
#

# %%
# %config InlineBackend.figure_formats = ["svg"]

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.optimize as opt
import scipy.stats as stats

# %%
rng = np.random.default_rng(42)

# %%
lh_pos_x, lh_pos_y = 0.0, 5.0


# %% [markdown]
# $$
# x_0 = x_l + y_l \tan(\phi)
# $$

# %%
def to_shoreline_coords(lh_x, lh_y, phi):
    return lh_x + lh_y * np.tan(phi)


# %%
phi = 0.86
x0 = to_shoreline_coords(lh_pos_x, lh_pos_y, phi)

fig, ax = plt.subplots(figsize=(6, 4))

shoreline = patches.Rectangle(
    (-10, -1),
    20,
    1,
    linewidth=1,
    edgecolor="k",
    facecolor="none",
    hatch="//",
    label="Shoreline",
)

ax.set_aspect(1)
ax.set_xlim(-10, 10)
ax.set_ylim(-0.5, 10.0)

ax.set_xlabel("Shoreline $x_0$")
ax.set_ylabel("Distance from shore $d$")

ax.add_patch(shoreline)
ax.plot([lh_pos_x, lh_pos_x], [0, lh_pos_y], ls="--", c="grey", zorder=0)

ax.annotate(
    "",
    xytext=(lh_pos_x, lh_pos_y),
    xy=(x0, 0),
    xycoords="data",
    arrowprops=dict(arrowstyle="->", color="y"),
    zorder=0,
)
ax.scatter(lh_pos_x, lh_pos_y, label="Lighthouse")
ax.text(0.45, 4.0, r"$\phi$", ha="center", va="center")
ax.legend()

# %%
phi_sample = rng.uniform(-np.pi / 2, np.pi / 2, size=200)
x_sample = to_shoreline_coords(lh_pos_x, lh_pos_y, phi_sample)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(x_sample, range=(-10, 10), bins=21)

ax.set_xlim(-10, 10)
ax.set_xlabel("Shoreline $x_0$")
ax.set_ylabel("Flashes observed")

# %% [markdown]
# Transformation of variable
# $$
# \phi \sim U\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)
# $$
# $$
# f(\phi) = \frac{1}{\pi} \quad \text{for} \quad \phi \in (-\pi/2,\pi/2) 
# $$
#
# $$
# \phi = \operatorname{atan}{\left(\frac{x_{0} - x_{l}}{y_{l}} \right)}
# $$
#
# $$
# f(x_0) = \frac{1}{\pi y_{l} \left[1 + \left(\frac{x_{0} - x_{l}}{y_{l}} \right)^2 \right]}
# $$
#
# $$
# \log(f(x_0)) = \log{\left(y_{l} \right)} - \log{\left(y_{l}^{2} + \left(x_{0} - x_{l}\right)^{2} \right)} - \log{\left(\pi \right)}
# $$

# %%
x_0 = sp.symbols("x_0", real=True)
x_l = sp.symbols("x_l", real=True)
y_l = sp.symbols("y_l", real=True, positive=True)
phi = sp.symbols("phi", real=True)

# %%
(phi_sol,) = sp.solve(sp.Eq(x_0, x_l + y_l * sp.tan(phi)), phi)
sp.Eq(phi, phi_sol)

# %%
# Lorentz/Cauchy distribution
fx = sp.Abs(sp.diff(phi_sol, x_0)) / sp.pi
fx

# %%
log_fx = sp.simplify(sp.log(fx))
log_fx


# %%
def pdf(x_0, x_l, y_l):
    return stats.cauchy.pdf(x_0, loc=x_l, scale=y_l)


def logpdf(x_0, x_l, y_l):
    return stats.cauchy.logpdf(x_0, loc=x_l, scale=y_l)


# %%
x_plot = np.linspace(-10, 10, 100)
density = (
    pdf(x_plot, lh_pos_x, lh_pos_y)
    / sp.integrate(fx, (x_0, -10, 10)).subs({x_l: lh_pos_x, y_l: lh_pos_y}).evalf()
)

fig, ax = plt.subplots()
ax.hist(x_sample, range=(-10, 10), bins=21, density=True)
ax.plot(x_plot, density)

ax.set_xlim(-10, 10)
ax.set_xlabel("Shoreline $x_0$")
ax.set_ylabel("Flashes observed")

# %%
res = opt.minimize(
    lambda p: -2 * logpdf(x_sample, p[0], p[1]).sum(),
    # Estimate good starting values for optimization
    [np.median(x_sample) + 0.1, stats.iqr(x_sample) / 2 + 0.1],
)
res

# %%
xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(0.1, 10, 100))

# %%
z = -2 * logpdf(x_sample.reshape(-1, 1, 1), xx, yy).sum(axis=0) - res.fun

# %%
levels = stats.chi2.ppf(np.array([0.0, 0.68, 0.95, 0.99]), df=2)

# %%
fig, ax = plt.subplots()

shoreline = patches.Rectangle(
    (-10, -1),
    20,
    1,
    linewidth=1,
    edgecolor="k",
    facecolor="none",
    hatch="//",
    label="Shoreline",
)

ax.set_aspect(1)
ax.set_xlim(-10, 10)
ax.set_ylim(-0.5, 10.0)
ax.set_xlabel("Shoreline $x_0$")
ax.set_ylabel("Distance from shore $d$")

ax.add_patch(shoreline)
ax.contourf(xx, yy, 2 * z, levels=levels)
ax.scatter(lh_pos_x, lh_pos_y, label="Lighthouse (actual)", c="r", marker=".")
ax.scatter(res.x[0], res.x[1], label="Lighthouse (estimate)", c="c", marker=".")
ax.legend()

x1, x2, y1, y2 = lh_pos_x - 1.0, lh_pos_x + 1.0, lh_pos_y - 1, lh_pos_y + 1
ax_ins = ax.inset_axes(
    [0.0, 0.5, 0.45, 0.45],
    xlim=(x1, x2),
    ylim=(y1, y2),
)
ax_ins.set_aspect(1)
ax_ins.contourf(
    xx,
    yy,
    2 * z,
    levels=levels,
)
ax_ins.scatter(lh_pos_x, lh_pos_y, c="r", marker=".")
ax_ins.scatter(res.x[0], res.x[1], c="c", marker=".")
ax.indicate_inset_zoom(ax_ins, ec="grey")

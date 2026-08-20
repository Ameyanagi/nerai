"""Generate the scipy curve_fit covariance cross-check fixture.

Run: uv run --with numpy --with scipy python scripts/generate_statistics_fixture.py
The printed values are pinned in tests/test_statistics.mojo.
"""

import numpy as np
from scipy.optimize import curve_fit

rng = np.random.default_rng(42)
t = np.linspace(0.0, 5.0, 25)
noise = rng.normal(0.0, 0.05, size=t.size)
y = 2.5 * np.exp(-0.7 * t) + noise


def model(t, amplitude, rate):
    return amplitude * np.exp(-rate * t)


popt, pcov = curve_fit(model, t, y, p0=[1.0, 1.0])
residuals = model(t, *popt) - y
dof = t.size - 2
print("t =", [float(v) for v in t])
print("y =", [float(v) for v in y])
print("popt =", [float(v) for v in popt])
print("pcov =", [float(v) for v in pcov.ravel()])
print("perr =", [float(v) for v in np.sqrt(np.diag(pcov))])
print("reduced_chi2 =", float(residuals @ residuals) / dof)
print("dof =", dof)

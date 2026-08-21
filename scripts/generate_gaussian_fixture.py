"""Generate the scipy Gaussian-peak cross-check fixture.

Run: uv run --with numpy --with scipy python scripts/generate_gaussian_fixture.py
The printed values are pinned in tests/test_curve.mojo.
"""

import numpy as np
from scipy.optimize import curve_fit

rng = np.random.default_rng(7)
x = np.linspace(-5.0, 5.0, 41)


def model(x, amplitude, center, width):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


y = model(x, 3.0, 0.5, 1.2) + rng.normal(0.0, 0.08, x.size)
popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.0, 2.0])
print("x =", [float(v) for v in x])
print("y =", [float(v) for v in y])
print("popt =", [float(v) for v in popt])
print("perr =", [float(v) for v in np.sqrt(np.diag(pcov))])

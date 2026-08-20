"""Generate the bounded-vs-unbounded scipy cross-check fixture.

Run: uv run --with numpy --with scipy python scripts/generate_bounds_fixture.py
The printed values are pinned in tests/test_bounds.mojo.
"""

import numpy as np
from scipy.optimize import curve_fit

rng = np.random.default_rng(11)
t = np.linspace(0.0, 4.0, 12)
y = 0.5 + 0.04 * t + rng.normal(0.0, 0.02, t.size)


def model(t, amplitude, rate):
    return amplitude * np.exp(-rate * t)


unbounded, _ = curve_fit(model, t, y, p0=[0.5, 0.1], maxfev=20000)
bounded, _ = curve_fit(
    model, t, y, p0=[0.5, 0.1], bounds=([0.0, 0.0], [np.inf, np.inf])
)
print("t =", [float(v) for v in t])
print("y =", [float(v) for v in y])
print("unbounded popt =", [float(v) for v in unbounded])
print("bounded popt =", [float(v) for v in bounded])

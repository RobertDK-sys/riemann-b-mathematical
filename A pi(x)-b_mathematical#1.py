"""
Reconstruction of the prime-counting function pi(x) from the imaginary
parts b of the non-trivial zeros of the zeta function.

OPTIMIZED VERSION - WITH FILE LOGGING
--------------------------------------------------------------
VARIANT: positive AND negative b values are kept exactly as produced
by the equation (no b>0 filter, no forced (b,-b) pairing). The negative
values coming out of the calculation are independent solutions of the
zeta function, not artificial duplicates of positive b values.

Consequence for the parameters:
  n_max = t_max * (2 * k_range)
  n_points = n_max
--------------------------------------------------------------

  - detection of successive intersection points between the "exact
    pi(x)" curve (step function) and the "Reconstruction" curve
    (continuous)
  - these points are recorded in a DEDICATED, DISTINCT log file
    (intersections_pm.log), with the same "immediate flush" mechanism
    to survive a crash

_get_s2(t) systematically solves the characteristic equation via brentq,
for all t >= 1. The precomputed table _S2_TABLE (a shortcut for
t <= 100) has been removed.
"""

import math
import os
import gc
import traceback
import datetime
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import mpmath as mp

mp.mp.dps = 15

BASE_DIR = "/storage/emulated/0/A_algorithms-Python"
OUT_DIR = os.path.join(BASE_DIR, "outputs3")
LOG_PATH = os.path.join(BASE_DIR, "journal_execution.log")
INTERSECT_LOG_PATH = os.path.join(BASE_DIR, "intersections_pm.log")

_log_file = None
_intersect_file = None


def init_log():
    global _log_file
    os.makedirs(BASE_DIR, exist_ok=True)
    _log_file = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    log("=" * 60)
    log("New run (explicit b AND -b) started")


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    if _log_file is not None:
        try:
            _log_file.write(line + "\n")
            _log_file.flush()
        except Exception:
            pass


def init_intersect_log():
    global _intersect_file
    os.makedirs(BASE_DIR, exist_ok=True)
    _intersect_file = open(INTERSECT_LOG_PATH, "a", encoding="utf-8", buffering=1)
    log_intersect("=" * 60)
    log_intersect("New run (explicit b AND -b) - intersections")


def log_intersect(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    if _intersect_file is not None:
        try:
            _intersect_file.write(line + "\n")
            _intersect_file.flush()
        except Exception:
            pass


def mobius(n):
    if n == 1:
        return 1
    result = 1
    n0 = n
    p = 2
    while p * p <= n0:
        if n0 % p == 0:
            n0 //= p
            if n0 % p == 0:
                return 0
            result = -result
        p += 1
    if n0 > 1:
        result = -result
    return result


def li_power_mp(log_x, rho):
    return mp.ei(rho * log_x)


def J_from_zeros(x, b_values):
    """
    b_values contains the positive AND negative b values exactly as
    produced by the equation (independent solutions, not forced pairs).
    Each b, regardless of sign, counts for only ONE term (rho = 1/2 + i*b),
    so there is no factor of 2 here. The constant term -ln(2) stays whole
    (it does not depend on the number of zeros used).
    """
    x = float(x)
    if x <= 1:
        return 0.0
    log_x = mp.log(x)
    total = float(mp.ei(log_x))
    for b in b_values:
        rho = mp.mpc(0.5, b)
        total -= float(li_power_mp(log_x, rho).real)
    total -= math.log(2)
    return total


def pi_reconstruction(x, b_values, n_max):
    x = float(x)
    s = 0.0
    for n in range(1, n_max + 1):
        root = x ** (1.0 / n)
        if root < 2:
            break
        mu = mobius(n)
        if mu != 0:
            s += mu / n * J_from_zeros(root, b_values)
    return s


def pi_exact_step(xs, xmax):
    xmax = int(np.ceil(xmax))
    sieve = np.ones(xmax + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(xmax ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p:: p] = False
    primes = np.nonzero(sieve)[0]
    return np.searchsorted(primes, xs, side="right")


def _equation(s2, k3):
    A = 8 * s2 ** 6 - 24 * s2 ** 4 + 24 * s2 ** 2 - 8
    return abs(A) ** (2 / 3) * (24 * k3 ** 2 * s2 ** 6 - 72 * k3 ** 2 * s2 ** 4 - 24 * k3 ** 2) + 4 * s2 ** 4


def _get_s2(t):
    """
    Numerically solves the characteristic equation for k3 = -t and
    returns the root s2(t) in (-1, 0), for any t >= 1. This is the
    only place -- and the sole place -- where the solving happens.
    """
    return brentq(_equation, -0.9999, -0.0001, args=(-t,), xtol=1e-10)


def get_zero_imag_parts_custom(n_zeros, t_max=35, k_range=15):
    """
    Keeps the b values exactly as produced by the equation, positive AND
    negative, with no sign filter and no artificial pairing. A negative
    b coming out of the calculation is an independent solution, not the
    mirror of a positive b. Simple deduplication on the value (avoids
    near-numerical duplicates), sorted by increasing absolute value to
    favor the "closest" zeros first.
    """
    raw = []
    for t in range(1, t_max + 1):
        s2_val = _get_s2(t)
        for k in range(-k_range, k_range + 1):
            if k != 0:
                b = (math.asin(s2_val) - 2 * k * math.pi) / math.log(2)
                raw.append(b)

    raw_sorted = sorted(raw, key=abs)
    deduped = []
    for b in raw_sorted:
        if not any(abs(b - d) < 1e-5 for d in deduped):
            deduped.append(b)
        if len(deduped) >= n_zeros:
            break

    return deduped[:n_zeros]


def find_intersections(xs, exact, recon):
    """
    Detects successive intersection points between the step curve
    'exact' (integer values, discontinuous) and the continuous curve
    'recon'.

    Method: the sign of diff = recon - exact is observed between each
    pair of consecutive points on the xs grid. A sign change (or an
    exact zero value) indicates a crossing. The intersection point is
    estimated by linear interpolation of diff between the two abscissas,
    which gives a reasonable approximation of x_intersection even though
    'exact' is a step function (the resulting error is bounded by the
    step size of the xs grid).

    Returns a list of (x_intersection, y_intersection) tuples.
    """
    intersections = []
    diff = [r - e for r, e in zip(recon, exact)]

    for i in range(1, len(xs)):
        d0, d1 = diff[i - 1], diff[i]

        if d0 == 0:
            intersections.append((float(xs[i - 1]), float(exact[i - 1])))
            continue

        if d0 * d1 < 0:
            x0, x1 = float(xs[i - 1]), float(xs[i])
            t = d0 / (d0 - d1)
            x_cross = x0 + t * (x1 - x0)
            y0, y1 = float(exact[i - 1]), float(exact[i])
            y_cross = y0 + t * (y1 - y0)
            intersections.append((x_cross, y_cross))

    return intersections


def plot_reconstruction(n_zeros, x_max, n_max, n_points, b_values, save_path):
    xs = np.linspace(2, x_max, n_points)
    exact = pi_exact_step(xs, x_max)
    recon = [pi_reconstruction(x, b_values, n_max) for x in xs]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.step(xs, exact, color="white", where="post", label="pi(x) exact")
    ax.plot(xs, recon, color="gray", alpha=0.85, label="Reconstruction")
    ax.set_title(f"Reconstruction ({n_zeros} zeros, signed b)", color="white")
    ax.tick_params(colors="white")
    if save_path:
        fig.savefig(save_path, dpi=100, facecolor="black")

    return fig, xs, exact, recon


def main():
    # n_max = t_max * (2 * k_range)  <-- positive and negative b already counted
    scenarios = [
        (100, 10, 5),
        (800, 20, 20),
        (20000, 100, 100),
    ]

    for n_zeros, t_max, k_range in scenarios:
        n_max = t_max * (2 * k_range)
        n_points = n_max

        log(f"Starting: N={n_zeros}, n_max={n_max}")
        b_values = get_zero_imag_parts_custom(n_zeros, t_max, k_range)
        save_path = os.path.join(OUT_DIR, f"recon_pm_{n_zeros}.png")

        fig, xs, exact, recon = plot_reconstruction(
            n_zeros, 80, n_max, n_points, b_values, save_path
        )
        plt.close(fig)
        gc.collect()

        if os.path.exists(save_path):
            log(f"  -> Saved: {save_path}")
        else:
            log(f"  -> FAILED to save {save_path}")

        # --- Detection and logging of intersections ---
        intersections = find_intersections(xs, exact, recon)
        log_intersect(f"Scenario N={n_zeros} (n_max={n_max}, x_max=80): "
                       f"{len(intersections)} intersection(s) found")
        for idx, (xc, yc) in enumerate(intersections, start=1):
            log_intersect(f"  N={n_zeros} | intersection #{idx}: "
                           f"x = {xc:.6f}, pi(x) ~ {yc:.6f}")


if __name__ == "__main__":
    init_log()
    init_intersect_log()
    try:
        main()
    except Exception:
        log(traceback.format_exc())
    finally:
        if _log_file:
            _log_file.close()
        if _intersect_file:
            _intersect_file.close()
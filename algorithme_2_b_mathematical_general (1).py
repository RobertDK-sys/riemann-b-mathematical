"""
Algorithme 2 — Résolution générale de l'équation caractéristique
--------------------------------------------------------------------
_get_s2(t) appelle brentq(_equation, ...) pour obtenir la racine réelle s2(t) ∈ [-0.9999,
-0.0001], pour tout t ≥ 1 (et non plus seulement t = 1..100). _S2_TABLE
est construite en appelant _get_s2(t).

    0 = |8·s2^6 − 24·s2^4 + 24·s2^2 − 8|^(2/3)
        · (24·k3^2·s2^6 − 72·k3^2·s2^4 − 24·k3^2) + 4·s2^4,   avec k3 = -t

    b(t, k) = [arcsin(s2(t)) − 2·k·π] / ln(2),   k entier non nul
"""
from math import asin, log, pi
from scipy.optimize import brentq

t_max = 100
k_range = 10


def _equation(s2, k3):
    A = 8 * s2 ** 6 - 24 * s2 ** 4 + 24 * s2 ** 2 - 8
    return abs(A) ** (2 / 3) * (24 * k3 ** 2 * s2 ** 6 - 72 * k3 ** 2 * s2 ** 4 - 24 * k3 ** 2) + 4 * s2 ** 4


def _get_s2(t):
    """Résout numériquement l'équation caractéristique pour k3 = -t
    et renvoie la racine s2(t) dans (-1, 0). C'est ici — et seulement
    ici — que la résolution a lieu ; il n'y a plus de table stub."""
    return brentq(_equation, -0.9999, -0.0001, args=(-t,), xtol=1e-14)


# _S2_TABLE appelle désormais réellement _get_s2(t) pour chaque t,
# au lieu du placeholder linéaire de la version précédente.
_S2_TABLE = {t: _get_s2(t) for t in range(1, t_max + 1)}


if __name__ == "__main__":
    for t in range(1, t_max + 1):
        s2_val = _S2_TABLE[t]
        for signe in (1, -1):
            for k in range(-k_range, k_range + 1):
                if k != 0:
                    b = (asin(signe * abs(s2_val)) - 2 * k * pi) / log(2)
                    print('b =', b)

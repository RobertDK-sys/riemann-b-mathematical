"""
Algorithme 3 — Algorithme 2 + sommes partielles de zêta corrigées (Euler-Maclaurin)
Version ciblée autour de b = 14 :
    1) b = 14.0
    2) les 2 b_mathematical (issus de l'Algorithme 2) les plus proches de 14
    3) b_reference = 14.134725141734693790 (premier zéro non trivial de ζ)
------------------------------------------------------------------------------------
    S_n(s) = 1/1^s + ... + 1/n^s,   s = 1/2 + i·b

    ζ_M(s; n) = S_n - n^(1-s)/(1-s) - n^(-s)/2
                + Σ_{j=1..M} [B_2j/(2j)!] · s(s+1)...(s+2j-2) · n^(-s-2j+1)

Valeur de référence ζ(s) : algorithme de Borwein (indépendant d'Euler-Maclaurin).
"""
import csv
import math
from fractions import Fraction
from math import asin, log, pi, factorial

import numpy as np
from scipy.optimize import brentq

# ----------------------------------------------------------------------------
# Algorithme 2 (inchangé)
# ----------------------------------------------------------------------------
t_max = 100
k_range = 10


def _equation(s2, k3):
    A = 8 * s2 ** 6 - 24 * s2 ** 4 + 24 * s2 ** 2 - 8
    return abs(A) ** (2 / 3) * (24 * k3 ** 2 * s2 ** 6 - 72 * k3 ** 2 * s2 ** 4 - 24 * k3 ** 2) + 4 * s2 ** 4


def _get_s2(t):
    """Résout numériquement l'équation caractéristique pour k3 = -t
    et renvoie la racine s2(t) dans (-1, 0)."""
    return brentq(_equation, -0.9999, -0.0001, args=(-t,), xtol=1e-14)


_S2_TABLE = {t: _get_s2(t) for t in range(1, t_max + 1)}


def b_algorithme_2():
    """Génère (t, signe, k, b) comme dans l'Algorithme 2."""
    for t in range(1, t_max + 1):
        s2_val = _S2_TABLE[t]
        for signe in (1, -1):
            for k in range(-k_range, k_range + 1):
                if k != 0:
                    yield t, signe, k, (asin(signe * abs(s2_val)) - 2 * k * pi) / log(2)


# ----------------------------------------------------------------------------
# Algorithme 3 : paramètres
# ----------------------------------------------------------------------------
N_LISTE = (1000, 10000, 100000, 1000000)
M_TERMES = 4               # nombre de termes de Bernoulli ajoutés (B2, B4, B6, B8)
B_CIBLE = 14.0             # valeur de b testée directement
N_PROCHES = 2              # nombre de b_mathematical les plus proches de B_CIBLE
B_REFERENCE = 14.134725141734693790   # premier zéro non trivial (valeur de référence)
FICHIER_CSV = "resultats_algorithme_3_b14.csv"


def nombres_bernoulli(m):
    B = [Fraction(0)] * (m + 1)
    B[0] = Fraction(1)
    for n in range(1, m + 1):
        B[n] = -sum(Fraction(factorial(n + 1), factorial(k) * factorial(n + 1 - k)) * B[k]
                    for k in range(n)) / (n + 1)
    return B


_BERN = nombres_bernoulli(2 * M_TERMES)

# ln k et k^(-1/2) calculés une seule fois pour n = 10^6
_N_MAX = max(N_LISTE)
_LN = np.log(np.arange(1, _N_MAX + 1, dtype=np.float64))
_POIDS = np.exp(-0.5 * _LN)


def sommes_partielles(b):
    """S_n(1/2 + i·b) pour n dans N_LISTE (somme en double précision)."""
    phase = b * _LN
    re = _POIDS * np.cos(phase)
    im = -_POIDS * np.sin(phase)
    return {n: complex(math.fsum(re[:n]), math.fsum(im[:n])) for n in N_LISTE}


def zeta_corrige(S_n, n, s, M):
    """ζ_M(s; n) = S_n - T1 - T2 + Σ_{j=1..M} B_2j/(2j)! · s(s+1)...(s+2j-2) · n^(-s-2j+1)."""
    ln_n = math.log(n)
    n_moins_s = np.exp(-s * ln_n)                   # n^(-s)
    val = S_n - n * n_moins_s / (1 - s) - n_moins_s / 2   # n^(1-s) = n · n^(-s)
    prod = 1 + 0j
    for j in range(1, M + 1):
        if j == 1:
            prod = s
        else:
            prod = prod * (s + (2 * j - 3)) * (s + (2 * j - 2))
        coef = float(_BERN[2 * j] / factorial(2 * j))
        val += coef * prod * n_moins_s / n ** (2 * j - 1)
    return complex(val)


def zeta_borwein(s, n=60):
    """Valeur de référence de ζ(s) par l'algorithme de Borwein (série η accélérée)."""
    dk, acc = [], Fraction(0)
    for i in range(n + 1):
        acc += Fraction(factorial(n + i - 1) * 4 ** i, factorial(n - i) * factorial(2 * i))
        dk.append(acc)
    dn = dk[n]
    tot = 0j
    for k in range(n):
        tot += (-1) ** k * float((dn - dk[k]) / dn) * np.exp(-s * math.log(k + 1))
    return complex(tot / (1 - 2 ** (1 - s)))


def choisir_b():
    """Renvoie la liste d'étiquettes (type, t, signe, k, b) à traiter."""
    candidats = sorted(b_algorithme_2(), key=lambda x: abs(x[3] - B_CIBLE))
    cibles = [("b=14.0", "", "", "", B_CIBLE)]
    for rang, (t, signe, k, b) in enumerate(candidats[:N_PROCHES], start=1):
        cibles.append((f"b_mathematical_{rang}", t, signe, k, b))
    cibles.append(("b_reference", "", "", "", B_REFERENCE))
    return cibles


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    lignes = []
    for etiquette, t, signe, k, b in choisir_b():
        s = complex(0.5, b)
        z_ref = zeta_borwein(s)
        S = sommes_partielles(b)
        for n in N_LISTE:
            z0 = zeta_corrige(S[n], n, s, 0)
            zM = zeta_corrige(S[n], n, s, M_TERMES)
            lignes.append({
                "type": etiquette, "t": t, "signe": signe, "k": k, "b": b, "n": n,
                "S_n_re": S[n].real, "S_n_im": S[n].imag,
                "zeta_M0_re": z0.real, "zeta_M0_im": z0.imag,
                f"zeta_M{M_TERMES}_re": zM.real, f"zeta_M{M_TERMES}_im": zM.imag,
                "zeta_ref_re": z_ref.real, "zeta_ref_im": z_ref.imag,
                "abs_zeta_ref": abs(z_ref),
                f"erreur_M{M_TERMES}": abs(zM - z_ref),
            })

    with open(FICHIER_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        w.writeheader()
        w.writerows(lignes)

    print(f"{len(lignes) // len(N_LISTE)} valeurs de b ; M = {M_TERMES} termes de correction ; "
          f"résultats complets -> {FICHIER_CSV}\n")
    for L in lignes:
        zM = complex(L[f"zeta_M{M_TERMES}_re"], L[f"zeta_M{M_TERMES}_im"])
        info = f"t={L['t']} signe={L['signe']} k={L['k']}" if L["t"] != "" else ""
        print(f"[{L['type']:<18}] b={L['b']:.18f} n={L['n']:>8} {info}\n"
              f"      zeta(s) ~ {zM.real:+.12f}{zM.imag:+.12f}i  "
              f"|zeta_ref|={L['abs_zeta_ref']:.3e}  |err|={L[f'erreur_M{M_TERMES}']:.1e}")

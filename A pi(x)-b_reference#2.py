"""
Reconstruction du comptage des nombres premiers pi(x) a partir des parties
imaginaires b des zeros non triviaux de la fonction zeta.

Principe (formule explicite de Riemann) :

  J(x) = li(x) - sum_rho li(x^rho) - ln(2) + integrale correctrice (negligeable ici)

  pi(x) = sum_{n=1}^{inf} mu(n)/n * J(x^(1/n))      (inversion de Mobius)

Les zeros rho = 1/2 + i*b viennent par paires conjuguees (1/2+ib, 1/2-ib),
donc chaque b > 0 contribue pour -2*Re(li(x^rho)) dans J(x).

Plus on ajoute de zeros (b_1, b_2, ...), plus la courbe grise "reconstruite"
epouse la marche d'escalier blanche de pi(x) exact.

CORRECTIF (bug de phase) :
---------------------------
La version precedente calculait li(x^rho) en passant par z = x**rho (mpmath),
qui repasse ensuite par log(z) en interne. Or log() ramene toujours la partie
imaginaire dans (-pi, pi] : pour un b eleve, rho*log(x) peut valoir plusieurs
centaines (ex. 308.9 pour b~800 et x=90), et ce "nombre de tours" est perdu
des qu'on repasse par x**rho puis log(...). Le resultat redevient une valeur
quelconque dans (-pi, pi], ce qui fait diverger completement la reconstruction
des qu'on ajoute des zeros a b eleve (courbe qui plonge vers -250 au lieu de
suivre pi(x) exact).

Correction : on calcule directement Ei(rho * log(x)) via mpmath.ei, sans
jamais repasser par x**rho ni par un second log(). C'est a la fois plus
rapide (une seule evaluation de log par x) et mathematiquement correct,
puisque li(x^rho) = Ei(log(x^rho)) = Ei(rho * log(x)) par definition, sans
ambiguite de branche a lever ici.

MODIFICATION : les images sont enregistrees dans le meme repertoire de
sortie que l'algorithme_2 (OUT_DIR ci-dessous), pour faciliter la comparaison
des courbes des deux scripts.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpf, mpc, li, log, ei, zetazero

mp.dps = 25  # precision decimale

# ---------------------------------------------------------------------------
# 0) Repertoire de sortie commun avec algorithme_2
# ---------------------------------------------------------------------------
BASE_DIR = "/storage/emulated/0/A_algorithms-Python"
OUT_DIR = os.path.join(BASE_DIR, "outputs3")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1) Fonction de Mobius (pour l'inversion)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2) J(x) tronque a N zeros (b_values = liste des parties imaginaires)
#    CORRIGE : Ei(rho*log(x)) direct, plus de x**rho / log() intermediaire
# ---------------------------------------------------------------------------
def J_from_zeros(x, b_values):
    x = mpf(x)
    if x <= 1:
        return mpf(0)
    log_x = log(x)          # calcule une seule fois, reutilise pour tous les zeros
    total = li(x)            # li(x) reel : aucun probleme de branche ici
    for b in b_values:
        rho = mpc(0.5, b)
        total -= 2 * ei(rho * log_x).real   # <-- correction du bug de phase
    total -= log(2)
    return total


# ---------------------------------------------------------------------------
# 3) pi(x) reconstruit par inversion de Mobius de J
# ---------------------------------------------------------------------------
def pi_reconstruction(x, b_values, n_max=25):
    x = mpf(x)
    s = mpf(0)
    n = 1
    while n <= n_max:
        root = x ** (mpf(1) / n)
        if root < 2:
            break
        mu = mobius(n)
        if mu != 0:
            s += mpf(mu) / n * J_from_zeros(root, b_values)
        n += 1
    return s


# ---------------------------------------------------------------------------
# 4) pi(x) exact (marche d'escalier), via crible simple
# ---------------------------------------------------------------------------
def pi_exact_step(xs, xmax):
    xmax = int(np.ceil(xmax))
    sieve = np.ones(xmax + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(xmax ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p:: p] = False
    primes = np.nonzero(sieve)[0]
    return np.searchsorted(primes, xs, side="right")


# ---------------------------------------------------------------------------
# 5) Recuperation des N premiers zeros de zeta (calcules une seule fois)
# ---------------------------------------------------------------------------
def get_zero_imag_parts(n_zeros):
    return [float(zetazero(k).imag) for k in range(1, n_zeros + 1)]


# ---------------------------------------------------------------------------
# 6) Trace pour un nombre de zeros donne
# ---------------------------------------------------------------------------
def plot_reconstruction(n_zeros, x_max=100, n_points=150, b_values=None,
                         save_path=None):
    if b_values is None:
        b_values = get_zero_imag_parts(n_zeros)
    else:
        b_values = b_values[:n_zeros]

    xs = np.linspace(2, x_max, n_points)
    exact = pi_exact_step(xs, x_max)
    recon = [float(pi_reconstruction(x, b_values)) for x in xs]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.step(xs, exact, color="white", where="post", linewidth=1.5, label="pi(x) exact")
    ax.plot(xs, recon, color="gray", linewidth=1.2, alpha=0.85, label="Reconstruction")

    ax.set_title("Prime-counting reconstruction", color="white")
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("pi(x)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(alpha=0.15)

    ax.text(0.03, 0.92, f"zeros passed: {n_zeros}", transform=ax.transAxes,
            color="white", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="black", edgecolor="cornflowerblue"))

    if save_path:
        fig.savefig(save_path, dpi=130, facecolor="black")
    return fig, ax


if __name__ == "__main__":
    # Exemple : calcule une fois les N premiers zeros, puis trace
    # plusieurs etapes de reconstruction (comme dans l'animation de reference).
    N_MAX = 79  # on couvre bien la plus grande etape demandee (79)
    print("Calcul des zeros de zeta (peut prendre un moment)...")
    b_values = get_zero_imag_parts(N_MAX)
    print("Premiers b (parties imaginaires) :", [round(b, 4) for b in b_values[:5]], "...")

    for n in [16, 46, 79]:
        save_path = os.path.join(OUT_DIR, f"recon_{n}.png")
        print(f"Generation du graphe pour {n} zeros...")
        fig, ax = plot_reconstruction(n, x_max=100, n_points=120,
                                       b_values=b_values,
                                       save_path=save_path)
        plt.close(fig)
        print(f"Image enregistree : {save_path}")

    print("Images generees dans :", OUT_DIR)

"""Corrections for the S and P states of positronium up to O(m alpha^6)."""

import os
import pkg_resources
import numpy as np
from math import log, pi
from scipy.special import psi, zeta
from .constants import alpha, gamma_e

DATA_PATH = pkg_resources.resource_filename("pyps", "data/")

# Bethe logarithm data
try:
    log_k0_fil = os.path.join(DATA_PATH, "log_k0_data.npy")
    log_k0_data = np.load(log_k0_fil)
except:
    log_k0_fil = os.path.join(DATA_PATH, "log_k0_data.dat")
    log_k0_data = np.loadtxt(log_k0_fil)


def delta(a, b):
    """Kronecker-delta function.

    Parameters
    ----------
    a : int
    b : int

    Returns
    -------
    1 if a == b else 0
    """
    return int(a == b)


def log_k0(n, l):
    """Bethe logarithms for S and P Rydberg States.

    Parameters
    ----------
    n : int [1... 200]
    l : int [0, 1]

    Returns
    -------
    float

    Reference
    ---------
    Bethe Logarithms for Rydberg States: Numerical Values for n ≤ 200
    Ulrich D. Jentschura and Peter J. Mohr (2005)

    https://arxiv.org/abs/quant-ph/0504002v1
    """
    n = min(n, 200)
    return log_k0_data[n, l]


def energy_average(n):
    """Spin-averaged energy of the S states.

    Parameters
    ----------
    n : int

    Returns
    -------
    float [au]

    Reference
    ---------
    Positronium S state spectrum: analytic results at O(m alpha^6)
    Andrzej Czarnecki, Kirill Melnikov and Alexander Yelkhovsky (1999)

    https://arxiv.org/abs/hep-ph/9901394v2

    Eqs. 92 & 95
    """
    return (
        -1.0 / (4.0 * n**2.0)
        + alpha**2.0 / (16.0 * n**3.0) * (11.0 / (4.0 * n) - 1.0)
        + alpha**3.0
        / (8.0 * pi * n**3.0)
        * (
            -6.0 * log(alpha)
            - (16.0 / 3.0) * log_k0(n, 0)
            + (14.0 / 3.0) * (log(4.0 / n) + psi(n) + gamma_e)
            - 37.0 / 45.0
            - 3.0 * log(2.0)
            + 7.0 / (3.0 * n)
        )
        + alpha**4.0
        / (32.0 * n**3.0)
        * (
            -log(alpha / n)
            - psi(n)
            - gamma_e
            + (141.0 / 4.0) * (zeta(3.0) / (pi**2.0))
            + ((137.0 / 6.0) - (68.0 / (pi**2.0))) * log(2.0)
            + 1421.0 / (27.0 * pi**2.0)
            - 2435.0 / 432.0
            - 7.0 / n
            + 17.0 / (12.0 * n**2.0)
            - 69.0 / (16.0 * n**3.0)
        )
        - alpha**5.0 * (log(alpha)) ** 2.0 / (32.0 * pi * n**3.0) * (499.0 / 15.0)
    )


def energy_hfs(n):
    """Hyperfine splitting of the S states.

    Parameters
    ----------
    n : int

    Returns
    -------
    float [au]

    Reference
    ---------
    Positronium S state spectrum: analytic results at O(m alpha^6)
    Andrzej Czarnecki, Kirill Melnikov and Alexander Yelkhovsky (1999)

    https://arxiv.org/abs/hep-ph/9901394v2

    Eqs. 93 & 95
    """
    return (
        (7.0 / 12.0) * alpha**2.0 / n**3.0
        - (alpha**3.0) / (pi * n**3.0) * (8.0 / 9.0 + 0.5 * log(2.0))
        + alpha**4.0
        / n**3.0
        * (
            -5.0 / 24.0 * (log(alpha / n) + psi(n) + gamma_e)
            + 1367.0 / (648 * pi**2.0)
            - 4297.0 / 3456.0
            + (221.0 / 144.0 + 1.0 / (2.0 * pi**2.0)) * log(2.0)
            - 53.0 * zeta(3.0) / (32.0 * pi**2.0)
            + 5.0 / (8.0 * n)
            - 85.0 / (96.0 * n**2.0)
        )
        - alpha**5.0 * (log(alpha)) ** 2.0 / (32.0 * pi * n**3.0) * 28.0
    )


def energy_s(n, J):
    """Energy of the S states.

    Parameters
    ----------
    n : int
    J : int

    Returns
    -------
    float [au]

    Reference
    ---------
    Positronium S state spectrum: analytic results at O(m alpha^6)
    Andrzej Czarnecki, Kirill Melnikov and Alexander Yelkhovsky (1999)

    https://arxiv.org/abs/hep-ph/9901394v2

    Eq. 91
    """
    return energy_average(n) + (0.25 - delta(J, 0)) * energy_hfs(n)


def energy_3p2(n):
    """Energy of the triplet P states (J=2).

    Parameters
    ----------
    n : int

    Returns
    -------
    float [au]

    Reference
    ---------
    Positronium S state spectrum: analytic results at O(m alpha^6)
    Andrzej Czarnecki, Kirill Melnikov and Alexander Yelkhovsky (1999)

    https://arxiv.org/abs/hep-ph/9901394v2

    Eq. C1
    """
    return (
        -1.0 / (4.0 * n**2.0)
        - alpha**2.0 / (4.0 * n**3.0) * (13.0 / 30.0 - 11.0 / (16.0 * n))
        - alpha**3.0
        / (8.0 * pi * n**3.0)
        * (4.0 / 45.0 + 16.0 / 3.0 * log_k0(n, 1))
        + alpha**4.0
        / n**3.0
        * (
            -69.0 / (512.0 * n**3.0)
            + 559.0 / (4800.0 * n**2.0)
            - 169.0 / (4800.0 * n)
            + 20677.0 / 432000.0
            - 3.0 * log(2.0) / 80.0
            + 9.0 * zeta(3.0) / (160.0 * pi**2.0)
            + 13.0 / (128.0 * pi**2.0)
        )
    )


def energy_3p1(n):
    """Energy of the triplet P states (J=1).

    Parameters
    ----------
    n : int

    Returns
    -------
    float [au]

    Reference
    ---------
    Positronium S state spectrum: analytic results at O(m alpha^6)
    Andrzej Czarnecki, Kirill Melnikov and Alexander Yelkhovsky (1999)

    https://arxiv.org/abs/hep-ph/9901394v2

    Eq. C1
    """
    return (
        -1.0 / (4.0 * n**2.0)
        - alpha**2.0 / (4.0 * n**3.0) * (5.0 / 6.0 - 11.0 / (16.0 * n))
        - alpha**3.0 / (8.0 * pi * n**3.0) * (5.0 / 9.0 + 16.0 / 3.0 * log_k0(n, 1))
        + alpha**4.0
        / n**3.0
        * (
            -69.0 / (512.0 * n**3.0)
            + 77.0 / (320.0 * n**2.0)
            - 25.0 / (192.0 * n)
            + 493.0 / 17280.0
            + log(2.0) / 48.0
            - zeta(3.0) / (32.0 * pi**2.0)
            - 179.0 / (3456.0 * pi**2.0)
        )
    )


def energy_3p0(n):
    """Energy of the triplet P states (J=0).

    Parameters
    ----------
    n : int

    Returns
    -------
    float [au]

    Reference
    ---------
    Positronium S state spectrum: analytic results at O(m alpha^6)
    Andrzej Czarnecki, Kirill Melnikov and Alexander Yelkhovsky (1999)

    https://arxiv.org/abs/hep-ph/9901394v2

    Eq. C1
    """
    return (
        -1.0 / (4.0 * n**2.0)
        - alpha**2.0 / (4.0 * n**3.0) * (4.0 / 3.0 - 11.0 / (16.0 * n))
        - alpha**3.0
        / (8.0 * pi * n**3.0)
        * (25.0 / 18.0 + 16.0 / 3.0 * log_k0(n, 1))
        + alpha**4.0
        / n**3.0
        * (
            -69.0 / (512.0 * n**3.0)
            + 119.0 / (240.0 * n**2.0)
            - 1.0 / (3.0 * n)
            - 923.0 / 4320.0
            + log(2.0) / 8.0
            - 3.0 * zeta(3.0) / (16.0 * pi**2.0)
            - 203.0 / (576.0 * pi**2.0)
        )
    )


def energy_1p1(n):
    """Energy of the singlet P states (J=1).

    Parameters
    ----------
    n : int

    Returns
    -------
    float [au]

    Reference
    ---------
    Positronium S state spectrum: analytic results at O(m alpha^6)
    Andrzej Czarnecki, Kirill Melnikov and Alexander Yelkhovsky (1999)

    https://arxiv.org/abs/hep-ph/9901394v2

    Eq. C1
    """
    return (
        -1.0 / (4.0 * n**2.0)
        - alpha**2.0 / (4.0 * n**3.0) * (2.0 / 3.0 - 11.0 / (16.0 * n))
        - alpha**3.0
        / (8.0 * pi * n**3.0)
        * (7.0 / 18.0 + 16.0 / 3.0 * log_k0(n, 1))
        + alpha**4.0
        / n**3.0
        * (
            -69.0 / (512.0 * n**3.0)
            + 23.0 / (120.0 * n**2.0)
            - 1.0 / (12.0 * n)
            + 163.0 / 4320.0
        )
    )

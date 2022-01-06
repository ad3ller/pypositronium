"""Zeeman effect for positronium."""

import numpy as np
from sympy.physics.wigner import wigner_3j, wigner_6j
from .constants import CACHE_MAXSIZE

# cache decorator
from functools import lru_cache


@lru_cache(CACHE_MAXSIZE)
def _zeeman(L, S1, J1, MJ1, S2, J2, MJ2):
    """Ps Zeeman interaction (cached function).

    Paramters
    ---------
    L : int
        orbital angular momentum
    S1 : int
        spin, state 1
    J1 : int
        total angular momentum, state 1
    MJ1 : int
        projection of the total angular momentum, state 1
    S2 : int
        spin, state 2
    J2 : int
        total angular momentum, state 2
    MJ2 : int
        projection of the total angular momentum, state 2

    Returns
    -------
    float

    References
    ----------
    C. D. Dermer and J. C. Weisheit (1989)
    Phys Rev A, 40, 5526. doi: 10.1103/physreva.40.5526. 

    """
    return float(
        (-1.0) ** (L + MJ1)
        * ((-1.0) ** (S1 + S2) - 1.0)
        * np.sqrt(3.0 * (2 * J2 + 1) * (2 * J1 + 1))
        * wigner_3j(J2, 1, J1, -MJ2, 0, MJ1)
        * wigner_6j(S2, L, J2, J1, 1, S1)
    )


def zeeman_interaction(state_1, state_2):
    """Zeeman interaction between two states.

    Paramters
    ---------
    state_1 : State
    state_2 : State

    Returns
    -------
    float

    """
    if (
        state_1.S != state_2.S
        and state_1.n == state_2.n
        and state_1.L == state_2.L
        and state_1.MJ == state_2.MJ
    ):
        return _zeeman(
            state_1.L,
            state_1.S,
            state_1.J,
            state_1.MJ,
            state_2.S,
            state_2.J,
            state_2.MJ,
        )
    return 0.0

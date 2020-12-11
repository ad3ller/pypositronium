"""Stark effect for positronium."""

import warnings
import numpy as np
from sympy import integrate, oo, var
from sympy.physics.hydrogen import R_nl
from sympy.physics.wigner import wigner_3j, wigner_6j
from .constants import mu_me

# cache decorator
try:
    from functools import cache
except ImportError:
    # python < 3.9
    from functools import lru_cache

    cache = lambda func: lru_cache(None)(func)

# optional import
try:
    from numerov import radial_integral as radial_numerov
except ImportError:
    radial_numerov = None


@cache
def _radial_integral(
    n1, l1, n2, l2, numerov=False, numerov_step=0.005, numerov_rmin=0.65
):
    """Calculate the radial integral (cached).

    Parameters
    ----------
    n1 : int
        principal quantum number, state 1
    l1 : int
        orbital angular momentum, state 1
    n2 : int
        principal quantum number, state 2
    l2 : int
        orbital angular momentum, state 2

    numerov=False : bool
    numerov_step=0.005 : float
    numerov_rmin=0.65 : float

    Returns
    -------
    float

    Examples
    --------
    >>> # sympy
    >>> radial_integral(10, 4, 11, 5)
    63.4960193562957

    >>> # numerov
    >>> radial_integral(10, 4, 11, 5, numerov=True)
    63.496017658724504

    Nb.  If numerov fails, automatically reverts to sympy

    """
    if numerov:
        ri = float(radial_numerov(n1, l1, n2, l2, step=numerov_step, rmin=numerov_rmin))
        if not np.isnan(ri):
            return ri
        else:
            # TODO fix numerov.radial_integral()
            warnings.warn(
                f"numerov.radial_integral returned nan for n1={n1}, l1={l1}, n2={n2}, l2={l2}.\\Fallback to scipy"
            )
    var("r")
    return float(
        integrate(R_nl(n1, l1, r) * r ** 3 * R_nl(n2, l2, r), (r, 0, oo)).evalf()
    )


@cache
def _ang_integral(S, L1, J1, MJ1, L2, J2, MJ2):
    """Calculate the angular integral (cached).

    Parameters
    ----------
    S : int
        spin
    L1 : int
        orbital angular momentum, state 1
    J1 : int
        total angular momentum, state 1
    MJ1 : int
        projection of the total angular momentum, state 1
    L2 : int
        orbital angular momentum, state 2
    J2 : int
        total angular momentum, state 2
    MJ2 : int
        projection of the total angular momentum, state 2

    Returns
    -------
    float

    """
    return float(
        (-1.0) ** (S + 1 + MJ2)
        * np.sqrt(max(L1, L2) * (2 * J2 + 1) * (2 * J1 + 1))
        * wigner_3j(J2, 1, J1, -MJ2, 0, MJ1)
        * wigner_6j(S, L2, J2, 1, J1, L1)
    )


def stark_interaction(state_1, state_2, numerov=False):
    """Stark interaction between two states.

    Paramters
    ---------
    state_1 : State
    state_2 : State
    numerov=False : bool
        use the numerov method?

    Returns
    -------
    float

    """
    if (
        abs(state_1.L - state_2.L) == 1
        and state_1.S == state_2.S
        and state_1.MJ == state_2.MJ
    ):
        return (
            _ang_integral(
                state_1.S,
                state_1.L,
                state_1.J,
                state_1.MJ,
                state_2.L,
                state_2.J,
                state_2.MJ,
            )
            * _radial_integral(
                state_1.n, state_1.L, state_2.n, state_2.L, numerov=numerov
            )
            / mu_me
        )
    return 0.0

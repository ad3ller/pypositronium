""" Stark effect
"""
import numpy as np
from sympy import integrate, oo, var
from sympy.physics.hydrogen import R_nl
from sympy.physics.wigner import wigner_3j, wigner_6j
from .constants import mu_me

# optional import
try:
    from numerov import radial_integral as radial_numerov
except ImportError:
    radial_numerov = None


def radial_integral(n1, l1, n2, l2,
                    numerov=False, numerov_step=0.005, numerov_rmin=0.65):
    """ Calculate the radial lintegral for two states of hydrogen.

    args:
        n1  :: int
        l1  :: int
        n2  :: int
        l2  :: int

    kwargs:
        numerov=False     :: bool
        numerov_step=0.01 :: float
        numerov_rmin=0.65 :: float

    return:
        float

    example:

    >>> # sympy
    >>> radial_integral(10, 4, 11, 5)
    63.4960193562957

    >>> # numerov
    >>> radial_integral(10, 4, 11, 5, numerov=True)
    63.496017658724504
    """
    if numerov:
        return radial_numerov(n1, l1, n2, l2,
                              step=numerov_step,
                              rmin=numerov_rmin)
    var("r")
    return float(integrate(R_nl(n1, l1, r) * r**3 * R_nl(n2, l2, r),
                           (r, 0, oo)).evalf())


def stark_interaction(state_1, state_2, numerov=False):
    """ Stark interaction between two states,

    return:
        ⟨ state_2 | r cos(theta) | state_1 ⟩
    """
    if abs(state_1.L - state_2.L) == 1 and state_1.S == state_2.S:
        return ((-1.0)**(state_1.S + 1 + state_2.MJ)
                * np.sqrt(max(state_1.L, state_2.L)
                          * (2*state_2.J + 1)
                          * (2*state_1.J + 1))
                * wigner_3j(state_2.J, 1, state_1.J,
                            -state_2.MJ, 0, state_1.MJ)
                * wigner_6j(state_2.S, state_2.L, state_2.J,
                            1, state_1.J, state_1.L)
                * radial_integral(state_1.n, state_1.L,
                                  state_2.n, state_2.L,
                                  numerov=numerov)
                / mu_me)
    return 0.0

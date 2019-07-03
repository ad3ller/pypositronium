""" Stark effect
"""
import numpy as np
from sympy.physics.wigner import wigner_3j, wigner_6j
from .constants import mu_me
from .numerov import rad_overlap


def stark_interaction(state_1, state_2):
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
                * rad_overlap(state_1.n, state_1.L,
                              state_2.n, state_2.L)
                / mu_me)
    return 0.0

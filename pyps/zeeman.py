""" Zeeman effect
"""
import numpy as np
from sympy.physics.wigner import wigner_3j, wigner_6j


def zeeman_interaction(state_1, state_2):
    """ Zeeman interaction between two states,

        returns:
            ⟨ state_2 | Hz | state_1 ⟩
    """
    if state_1.L == state_2.L:
        return ((-1.0)**(state_1.L + state_1.MJ)
                * ((-1.0)**(state_1.S + state_2.S) - 1.0)
                * np.sqrt(3.0 * (2*state_2.J + 1) * (2*state_1.J + 1))
                * wigner_3j(state_2.J, 1, state_1.J,
                            -state_2.MJ, 0, state_1.MJ)
                * wigner_6j(state_2.S, state_2.L, state_2.J,
                            state_1.J, 1, state_1.S))
    return 0.0

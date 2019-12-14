""" positronium energy levels (approximate)
"""
from .constants import alpha, mu_me, atomic_units


def epsilon(state):
    """ scaling of the fine structure shift.
    """
    if state.S == 0:
        # singlet
        return 0.0
    elif state.S == 1:
        # triplet
        delta = int(state.L == 0)
        if state.J == state.L + 1:
            omega = (3*state.L + 4) / ((state.L + 1) * (2*state.L + 3))
        elif state.J == state.L:
            omega = -1 / (state.L * (state.L + 1))
        elif state.J == state.L - 1:
            omega = -(3*state.L - 1) / (state.L * (2*state.L - 1))
        else:
            raise ValueError("The total angular momentum 'J' must "
                             + "be in the range L - 1 < J < L + 1")
        return 7 / 6 * delta + (1 - delta) / (2 * (2*state.L + 1)) * omega
    else:
        raise ValueError("The total spin quantum number 'S' must be 0 or 1.")


def fine_structure(state):
    """ first-order fine structure shift for state.

        H. A. Bethe and E. E. Salpeter (1957)
        Quantum Mechanics of One- and Two-Electron Systems
    """
    return (11 / 32 * state.n**-4
            + (epsilon(state) - 1.0 / (2*state.L + 1))
            * 1.0 / (state.n**3.0)) * alpha**2.0


@atomic_units('energy')
def energy(state):
    """ energy, including fine structure.
    """
    return - mu_me * (0.5 / (state.n**2) - fine_structure(state))

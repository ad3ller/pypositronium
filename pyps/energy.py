"""Energy levels of positronium."""

from .constants import alpha, mu_me, atomic_units
from . import ma6


def _epsilon(L, S, J):
    """Scaling of the fine structure shift.

    Parameters
    ----------
    L : int
        orbital angular momentum
    S : int
        spin
    J : int
        total angular momentum

    Returns
    -------
    float
    """
    if S == 0:
        # singlet
        return 0.0
    elif S == 1:
        # triplet
        delta = int(L == 0)
        if J == L + 1:
            omega = (3 * L + 4.0) / ((L + 1) * (2 * L + 3))
        elif J == L:
            omega = -1.0 / (L * (L + 1))
        elif J == L - 1:
            omega = -(3 * L - 1.0) / (L * (2 * L - 1))
        else:
            raise ValueError(
                "The total angular momentum 'J' must "
                + "be in the range L - 1 < J < L + 1"
            )
        return 7.0 / 6.0 * delta + (1 - delta) / (2.0 * (2 * L + 1)) * omega
    else:
        raise ValueError("The total spin quantum number 'S' must be 0 or 1.")


def fine_structure(n, L, S, J):
    """First-order fine structure shift for state.

    Parameters
    ----------
    n : int
        principal quantum number
    L : int
        orbital angular momentum
    S : int
        spin
    J : int
        total angular momentum

    Returns
    -------
    float

    References
    ----------
    H. A. Bethe and E. E. Salpeter (1957)
    Quantum Mechanics of One- and Two-Electron Systems
    """
    return (
        11 / 32 * n**-4 + (_epsilon(L, S, J) - 1.0 / (2 * L + 1)) * 1.0 / (n**3.0)
    ) * alpha**2.0


@atomic_units("energy")
def energy(state, m_alpha6=True, **kwargs):
    """Field-free energy of state.

    Parameters
    ----------
    state : State
    m_alpha6 : bool (default=True)
        include terms up to the O(m alpha^6) for S and P states?
    units : str

    Returns
    -------
    energy : float
    """
    if m_alpha6:
        if state.L == 0:
            return ma6.energy_s(state.n, state.J)
        if state.L == 1:
            if state.S == 0:
                return ma6.energy_1p1(state.n)
            if state.J == 2:
                return ma6.energy_3p2(state.n)
            if state.J == 1:
                return ma6.energy_3p1(state.n)
            if state.J == 0:
                return ma6.energy_3p0(state.n)
    return -mu_me * (
        0.5 / (state.n**2) - fine_structure(state.n, state.L, state.S, state.J)
    )

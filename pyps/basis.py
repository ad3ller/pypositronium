"""Basis set of atomic states."""

from collections import UserList
from dataclasses import dataclass
import numpy as np
from .energy import energy


@dataclass
class State:
    """Dataclass to represent ❘n, L, S, J, MJ⟩."""

    n: int
    L: int
    S: int
    J: int
    MJ: int

    def __post_init__(self):
        """Check quantum state."""
        self.validate()

    def __str__(self):
        """Dirac ket notation for quantum state."""
        return f"❘{self.n} {self.L} {self.S} {self.J} {self.MJ}⟩"

    def validate(self):
        """Validate quantum numbers."""
        assert isinstance(self.n, int)
        assert self.n > 0
        assert isinstance(self.L, int)
        assert self.L < self.n
        assert isinstance(self.S, int)
        assert self.S in [0, 1]
        assert isinstance(self.J, int)
        if self.S == 0:
            assert self.J == self.L
        elif self.L == 0:
            assert self.J == self.S
        else:
            assert self.L - self.S <= self.J <= self.L + self.S
        assert isinstance(self.MJ, int)
        assert -self.J <= self.MJ <= self.J

    def energy(self, **kwargs):
        """Quantum state energy."""
        return energy(self, **kwargs)

    def ket(self, show_MJ=True):
        """String of the form |n, l, S, J, [MJ]⟩$.

        Parameters
        ----------
        show_MJ=True : bool, default=True

        Returns
        -------
        str

        """
        if show_MJ:
            return f"❘{self.n}, {self.L} {self.S}, {self.J}, {self.MJ}⟩"
        else:
            return f"❘{self.n}, {self.L}, {self.S}, {self.J}⟩"

    def tex(self, show_MJ=True):
        """Tex string of the form n^{2S + 1}L_{J} (M_J = {MJ}).

        Parameters
        ----------
        show_MJ=True : bool

        Returns
        -------
        str

        """
        L = "SPDFGHIKLMNOQRTUVWXYZ"[int(self.L % 22)]
        tex_str = f"{self.n}^{2* self.S + 1}{L}_{self.J}"
        if show_MJ:
            tex_str += f"\\,(M_J = {self.MJ})"
        return tex_str


class Basis(UserList):
    """A list of State instances.

    Class Methods
    -------------
    build(n_values, L_values, S_values, MJ_values)
        Build an [n,L,S,J,MJ] basis set.

    Attributes
    ----------
    data : list
    num_states : int

    Methods
    -------
    values(attribute, array=False)
        Attribute values of the basis.
    where(condition, array=False)
        Matching elements of the basis.
    el(condition)
        First matching element of basis.
    argwhere(condition, array=False)
        Matching indexes of the basis.
    ix(condition)
        First matching index of basis.
    extract_states(inds):
        Subset of the basis set.

    """

    def __init__(self, states):
        """List of basis states.

        Parameters
        ----------
        states : List of instances of State

        Returns
        -------
        Basis

        """
        super().__init__(states)

    @classmethod
    def build(
        cls,
        n_values=None,
        L_values=None,
        S_values=None,
        MJ_values=None,
        filter_function=None,
        sort_key=energy,
    ):
        """Build an [n,L,S,J,MJ] basis set.

        Parameters
        ----------
        n_values : Iterable
        L_values : Iterable (range(n) if None)
        S_values : Iterable ([0, 1] if None)
        MJ_values : Iterable ([-J, ... J] if None)
        filter_function : None or Function
        sort_key : None or Function [default: energy()]

        Returns
        -------
        Basis

        """
        states = generate_basis(n_values, L_values, S_values, MJ_values)
        if filter_function is not None:
            states = filter(filter_function, states)
        if sort_key is not None:
            states = sorted(states, key=sort_key)
        return cls(states)

    @property
    def num_states(self):
        """Size of the basis set."""
        return len(self.data)

    def values(self, attribute, array=False):
        """Attribute values for all elements in the basis.

        Parameters
        ----------
        attribute : str
            e.g., 'n' or 'J'.
        array : bool

        Returns
        -------
        generator or numpy.ndarray

        Examples
        --------
        >>> n_values = list(basis.values('n'))

        """
        if array:
            return np.array(list(self.values(attribute)))
        return (getattr(el, attribute) for el in self.data)

    def where(self, condition, array=False):
        """Elements where condition mapped to basis evaluates as True.

        Parameters
        ----------
        condition : Function
        array : bool

        Returns
        -------
        generator or numpy.ndarray

        """
        if array:
            return np.array(list(self.where(condition)))
        return (x for x in self.data if condition(x))

    def el(self, condition):
        """First element where condition mapped to basis evaluates as True.

        Parameters
        ----------
        condition :: Function

        Returns
        -------
        state : State

        """
        return next(self.where(condition))

    def argwhere(self, condition, array=False):
        """Indexes where condition mapped to basis evaluates as True.

        Parameters
        ----------
        condition :: Function
        array :: bool

        Returns
        -------
        inds : generator or numpy.ndarray

        """
        if array:
            return np.array(list(self.argwhere(condition)))
        return (i for i, x in enumerate(self.data) if condition(x))

    def ix(self, condition):
        """First index where condition mapped to basis evaluates as True.

        Parameters
        ----------
        condition :: Function

        Returns
        -------
        index : int

        """
        return next(self.argwhere(condition))

    def extract_states(self, inds):
        """Subset of the basis.

        Parameters
        ----------
        inds : list or integer
            The states that should be kept.

        Returns
        -------
        Basis
            A new instance of :class:`basis.Basis` that contains only the states
            corresponding to the indices in `inds`.

        """
        if isinstance(inds, int):
            inds = [inds]
        states = np.array(self.data)[inds]
        return Basis(states)


def generate_basis(n_values, L_values=None, S_values=None, MJ_values=None):
    """Generate instances of State.

    Parameters
    ----------
    n_values :: Iterable
    L_values :: None or Iterable (default: range(n))
    S_values :: None or Iterable (default: [0, 1])
    MJ_values :: None or Iterable (default: [-J, ... J])

    Yields
    ------
    State

    """
    if n_values is None:
        return []
    if L_values is None:
        L_values = range(max(n_values))
    if S_values is None:
        S_values = [0, 1]
    # principal quantum number, n.
    for n in n_values:
        # total orbital angular momentum quantum number, L.
        for L in L_values:
            if L >= n:
                break
            # total spin quantum number, S.
            for S in S_values:
                # total angular momentum quantum number, J.
                if L == 0:
                    J_values = [S]
                elif S == 0:
                    J_values = [L]
                else:
                    J_values = [L + S, L, L - S]
                for J in J_values:
                    # projection of the total angular momentum, MJ.
                    if MJ_values is None:
                        for m in range(-J, J + 1):
                            yield State(n, L, S, J, m)
                    else:
                        for MJ in MJ_values:
                            if MJ in range(-J, J + 1):
                                yield State(n, L, S, J, MJ)

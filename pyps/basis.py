""" basis set of atomic quantum states
"""
from collections import UserList
from dataclasses import dataclass
import numpy as np
from .energy import energy


@dataclass
class State:
    """ dataclass for the quantum state ❘n, L, S, J, MJ⟩
    """
    __slots__ = ['n', 'L', 'S', 'J', 'MJ']
    n: int
    L: int
    S: int
    J: int
    MJ: int

    def __post_init__(self):
        """ check quantum state """
        self.validate()

    def __str__(self):
        """ ket notation """
        return f"❘{self.n} {self.L} {self.S} {self.J} {self.MJ}⟩"

    def validate(self):
        """ validate quantum numbers """
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
        """ state energy """
        return energy(self, **kwargs)

    def tex(self, show_MJ=True):
        """ Tex string of the form $n^{2S + 1}L_{J} (M_J = {MJ})$

        args:
            show_MJ=True :: bool

        return:
            str
        """
        L = "SPDFGHIKLMNOQRTUVWXYZ"[int(self.L / 22)]
        tex_str = f"${self.n}^{2* self.S + 1}{L}_{self.J}"
        if show_MJ:
            tex_str += f'\\,(M_J = {self.MJ})$'
        else:
            tex_str += '$'
        return tex_str


class Basis(UserList):
    """ A UserList of States

    args:
        n_values :: Iterable
        L_values ::  Iterable (range(n) if None)
        S_values ::  Iterable ([0, 1] if None)
        MJ_values :: Iterable ([-J, ... J] if None)

        filter_function :: None or Function
        sort_key :: None or Function [default: energy()]

    attributes:
        data :: list
        num_states :: int

    methods:
        values
            attribute values of the basis
        where
            a subset of the basis
        argwhere
            indexes of a subset of the basis
    """
    def __init__(self, n_values, L_values=None, S_values=None, MJ_values=None,
                 filter_function=None, sort_key=energy):
        """ Initialise collections.UserList """
        basis = generate_basis(n_values, L_values, S_values, MJ_values)
        if filter_function is not None:
            basis = filter(filter_function, basis)
        if sort_key is not None:
            basis = sorted(basis, key=sort_key)
        super().__init__(basis)

    @property
    def num_states(self):
        """ size of the basis set """
        return len(self.data)

    def values(self, attribute, ndarray=False):
        """ Attribute values for all elements in the basis.

        args:
            attribute :: str           e.g., n or J.
            ndarray    :: bool

        return:
            generator or numpy.ndarray

        example:
            n_values = list(basis.values('n'))
        """
        if ndarray:
            return np.array(list(self.values(attribute)))
        return (getattr(el, attribute) for el in self.data)

    def where(self, function, ndarray=False):
        """ Elements where function mapped to basis evaluates as True.

        args:
            function :: function
            ndarray    :: bool

        return:
            generator or numpy.ndarray
        """
        if ndarray:
            return np.array(list(self.where(function)))
        return (x for x in self if function(x))

    def argwhere(self, function, ndarray=False):
        """ Indexes where function mapped to basis evaluates as True.

        args:
            function :: function

        return:
            generator or numpy.ndarray
        """
        if ndarray:
            return np.array(list(self.argwhere(function)))
        return (i for i, x in enumerate(self) if function(x))


def generate_basis(n_values, L_values=None, S_values=None, MJ_values=None):
    """ generate instances of State

    args:
        n_values :: Iterable
        L_values :: None or Iterable (default: range(n))
        S_values :: None or Iterable (default: [0, 1])
        MJ_values :: None or Iterable (default: [-J, ... J])

    return:
        generator
    """
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

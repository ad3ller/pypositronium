""" Hamiltonian for positronium in electric and magnetic fields
"""
from collections.abc import Iterable
import numpy as np
from tqdm import trange
from .energy import energy
from .stark import stark_interaction
from .zeeman import zeeman_interaction
from .constants import atomic_units, mu_B, En_h, e, a0


class Hamiltonian(object):
    """ The total Hamiltonian matrix.
    """
    def __init__(self, basis):
        self.basis = basis
        # cache
        self._e0 = None
        self._stark_matrix = None
        self._zeeman_matrix = None

    def e0(self, update=False):
        """ field-free energy [atomic units].
        """
        if self._e0 is None or update:
            self._e0 = np.array([energy(x) for x in self.basis])
        return self._e0

    def h0_matrix(self, update=False):
        """ field-free Hamiltonian matrix [atomic units].
        """
        return np.diag(self.e0(update=update))

    def stark_matrix(self, Fz=1.0, update=False, **kwargs):
        """ Stark interaction matrix.

        args:
            Fz :: float
                electric field [atomic units]
            update :: Bool
                [re-]calculate and update cache

        kwrargs:
            tqdm_kw :: dict

        return:
            np.ndarray()
        """
        if self._stark_matrix is None or update:
            tqdm_kw = kwargs.get("tqdm_kw", {})
            self._stark_matrix = np.zeros([self.basis.num_states,
                                           self.basis.num_states])
            for i in trange(self.basis.num_states,
                            desc="calculate Stark terms", **tqdm_kw):
                # off-diagonal elements only
                for j in range(i + 1, self.basis.num_states):
                    state_1, state_2 = self.basis[i], self.basis[j]
                    self._stark_matrix[i][j] = stark_interaction(state_1,
                                                                 state_2)
                    # assume matrix is symmetric
                    self._stark_matrix[j][i] = self._stark_matrix[i][j]
        return Fz * self._stark_matrix

    def zeeman_matrix(self, Bz=1.0, update=False, **kwargs):
        """ Zeeman interaction matrix.

        args:
            Bz :: float
                magnetic field [atomic units]
            update :: Bool
                [re-]calculate and update cache

        kwrargs:
            tqdm_kw :: dict

        return:
            np.ndarray()
        """
        if self._zeeman_matrix is None or update:
            tqdm_kw = kwargs.get("tqdm_kw", {})
            self._zeeman_matrix = np.zeros([self.basis.num_states,
                                            self.basis.num_states])
            for i in trange(self.basis.num_states,
                            desc="calculate Zeeman terms", **tqdm_kw):
                for j in range(i, self.basis.num_states):
                    state_1, state_2 = self.basis[i], self.basis[j]
                    self._zeeman_matrix[i][j] = zeeman_interaction(state_1,
                                                                   state_2)
                    # assume matrix is symmetric
                    if i != j:
                        self._zeeman_matrix[j][i] = self._zeeman_matrix[i][j]
        return Bz * self._zeeman_matrix

    @atomic_units("energy")
    def matrix(self, electric_field=None, magnetic_field=None, units=None, **kwargs):
        """ total Hamiltonian matrix.

        args:
            electric_field      :: float     [V / m]
            magnetic_field      :: float     [T]

        kwargs:
            units="atomic_units"

        return:
            np.ndarray()
        """
        mat = self.h0_matrix(**kwargs)
        if electric_field is not None:
            Fz = electric_field * e * a0 / En_h
            mat += self.stark_matrix(Fz, **kwargs)
        if magnetic_field is not None:
            Bz = magnetic_field * mu_B / En_h
            mat += self.zeeman_matrix(Bz, **kwargs)           
        return mat

    @atomic_units("energy")
    def eig(self, electric_field=None, magnetic_field=None, units=None, **kwargs):
        """ eigenvalues and eigenvectors of the total Hamiltonian.

        args:
<<<<<<< HEAD
            electric_field      :: float     [V / m]
            magnetic_field      :: float     [T]

        kwargs:
            units="atomic_units"
=======
            Fz :: float
                electric field [atomic units]
            Bz :: float
                magnetic field [atomic units]
            units :: str
                output units, e.g., "SI" (default: "atomic units")
>>>>>>> f3795e4deac312c97efa19cc9ed9dcca68ca8212

        return:
            eigenvalues, eigenvectors
        """
        return np.linalg.eigh(self.matrix(electric_field, magnetic_field, **kwargs))

    @atomic_units("energy")
    def eigvals(self, electric_field=None, magnetic_field=None, units=None, **kwargs):
        """ eigenvalues of the total Hamiltonian.

        args:
<<<<<<< HEAD
            electric_field      :: float     [V / m]
            magnetic_field      :: float     [T]

        kwargs:
            units="atomic_units"
=======
            Fz :: float
                electric field [atomic units]
            Bz :: float
                magnetic field [atomic units]
            units :: str
                output units, e.g., "SI" (default: "atomic units")
>>>>>>> f3795e4deac312c97efa19cc9ed9dcca68ca8212

        return:
            eigenvalues
        """
        return np.linalg.eigvalsh(self.matrix(electric_field, magnetic_field, **kwargs))

    @atomic_units("energy")
    def eigamp(self, elements, electric_field=None, magnetic_field=None, units=None, **kwargs):
        """ eigenvalues and sum(eigenvector[elements]^2) of the total
        Hamiltonian.

        args:
<<<<<<< HEAD
            elements            :: Iterable
            electric_field      :: float     [V / m]
            magnetic_field      :: float     [T]

        kwargs:
            units="atomic_units"
=======
            Fz :: float
                electric field [atomic units]
            Bz :: float
                magnetic field [atomic units]
            units :: str
                output units, e.g., "SI" (default: "atomic units")
>>>>>>> f3795e4deac312c97efa19cc9ed9dcca68ca8212

        return:
            eigenvalues, amplitudes
        """
        vals, vec = np.linalg.eigh(self.matrix(electric_field, magnetic_field, **kwargs))
        amp = np.sum(vec[elements]**2.0, axis=0)
        return vals, amp

    @atomic_units("energy")
    def stark_map(self, electric_field,
                  magnetic_field=None, elements=False, units=None, **kwargs):
        """ The eigenvalues of the Hamiltonian for a range of electric fields.

        If units is not specified, eigenvalues are returned in atomic units.

        args:
            electric_field          :: Iterable      [V / m]
            magnetic_field=None     :: Number        [T]
            elements=False
                    :: Boolean
                            if True return eigenvectors

                    :: list or np.ndarray
                            return the sum of the square of the specified
                            elements of the eigenvectors.

        kwargs:
            update=False
            units="atomic_units"
            tqdm_kw :: dict

        return:
            eigenvalues[, eigenvectors or amplitudes]

        Nb. A large map with eignvectors can take up a LOT of memory.
        """
        tqdm_kw = kwargs.get("tqdm_kw", {})
        update = kwargs.get("update", False)
        # initialise output arrays
        num_fields = len(electric_field)
        values = np.empty((num_fields,
                           self.basis.num_states), dtype=float)
        if isinstance(elements, Iterable):
            amplitudes = np.empty((num_fields,
                                   self.basis.num_states), dtype=float)
        elif elements:
            vectors = np.empty((num_fields,
                                self.basis.num_states,
                                self.basis.num_states), dtype=float)
        # field-free matrix
        base_matrix = self.h0_matrix()
        # magnetic_field
        if magnetic_field is not None:
            Bz = magnetic_field * mu_B / En_h
            base_matrix += self.zeeman_matrix(Bz=Bz, **kwargs)
        # update stark matrix
        if update or self._stark_matrix is None:
            self.stark_matrix(**kwargs)
        # loop over electric field values
        for i in trange(num_fields,
                        desc="diagonalise matrix", **tqdm_kw):
            Fz = electric_field[i] * e * a0 / En_h
            # diagonalise
            if isinstance(elements, Iterable):
                values[i], vec = np.linalg.eigh(base_matrix
                                                + self.stark_matrix(Fz=Fz))
                amplitudes[i] = np.sum(vec[elements]**2.0, axis=0)
            elif elements:
                values[i], vectors[i] = np.linalg.eigh(base_matrix
                                                       + self.stark_matrix(Fz=Fz))
            else:
                values[i] = np.linalg.eigvalsh(base_matrix
                                               + self.stark_matrix(Fz=Fz))
        # output
        if isinstance(elements, Iterable):
            return values, amplitudes
        elif elements:
            return values, vectors
        else:
            return values

    @atomic_units("energy")
    def zeeman_map(self, magnetic_field,
                   electric_field=None, elements=False, units=None, **kwargs):
        """ The eigenvalues of the Hamiltonian for a range of magnetic fields.

        If units is not specified, eigenvalues are returned in atomic units.

        args:
            magnetic_field          :: Iterable      [T]
            electric_field=None     :: Number        [V / m]
            elements=False
                    :: Boolean
                            if True return eigenvectors

                    :: list or np.ndarray
                            return the sum of the square of the specified
                            elements of the eigenvectors.

        kwargs:
            update=False
            units="atomic_units"
            tqdm_kw :: dict

        return:
            eigenvalues [, eigenvectors or amplitudes]

        Nb. A large map with eignvectors can take up a LOT of memory.
        """
        tqdm_kw = kwargs.get("tqdm_kw", {})
        update = kwargs.get("update", False)
        # initialise output arrays
        num_fields = len(magnetic_field)
        values = np.empty((num_fields,
                           self.basis.num_states), dtype=float)
        if isinstance(elements, Iterable):
            amplitudes = np.empty((num_fields,
                                   self.basis.num_states), dtype=float)
        elif elements:
            vectors = np.empty((num_fields,
                                self.basis.num_states,
                                self.basis.num_states), dtype=float)
        # field-free matrix
        base_matrix = self.h0_matrix()
        # electric_field
        if electric_field is not None:
            Fz = electric_field * e * a0 / En_h
            base_matrix += self.stark_matrix(Fz=Fz, **kwargs)
        # update zeeman matrix
        if update or self._zeeman_matrix is None:
            self.zeeman_matrix(**kwargs)
        # loop over electric field values
        for i in trange(num_fields,
                        desc="diagonalise matrix", **tqdm_kw):
            Bz = magnetic_field[i] * mu_B / En_h
            # diagonalise
            if isinstance(elements, Iterable):
                values[i], vec = np.linalg.eigh(base_matrix
                                                + self.zeeman_matrix(Bz=Bz))
                amplitudes[i] = np.sum(vec[elements]**2.0, axis=0)
            elif elements:
                values[i], vectors[i] = np.linalg.eigh(base_matrix
                                                       + self.zeeman_matrix(Bz=Bz))
            else:
                values[i] = np.linalg.eigvalsh(base_matrix
                                               + self.zeeman_matrix(Bz=Bz))
        # output
        if isinstance(elements, Iterable):
            return values, amplitudes
        elif elements:
            return values, vectors
        else:
            return values

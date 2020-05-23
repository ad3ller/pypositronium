""" Hamiltonian for positronium in electric and magnetic fields
"""
from collections.abc import Iterable
import numpy as np
import scipy.sparse as sp
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
        self.dims = (self.basis.num_states,
                     self.basis.num_states)
        # cache
        self._h0_matrix = None
        self._stark_matrix = None
        self._zeeman_matrix = None

    @property
    def e0(self):
        """ field-free energy [atomic units].
        """
        return np.array([energy(x) for x in self.basis])

    def h0_matrix(self, dense=False, update=False):
        """ field-free Hamiltonian matrix [atomic units].

        args:
            dense :: Bool
                dense matrix?
            update :: Bool
                recalculate?
        
        return:
            scipy.sparse.dia_matrix
        """
        if self._h0_matrix is None or update:
            en = self.e0
            self._h0_matrix = sp.dia_matrix((en, 0),
                                            shape=self.dims, dtype=float)
        if dense:
            return self._h0_matrix.todense()
        return self._h0_matrix

    def stark_matrix(self, Fz=1.0, dense=False, update=False, **kwargs):
        """ Stark interaction matrix [atomic units].

        args:
            Fz :: float
                electric field [atomic units]
            dense :: Bool
                dense matrix?
            update :: Bool
                recalculate?

        kwrargs:
            tqdm_kw={}    :: dict
            numerov=False :: bool

        return:
            scipy.sparse.csr_matrix
        """
        if self._stark_matrix is None or update:
            tqdm_kw = kwargs.get("tqdm_kw", {})
            numerov = kwargs.get("numerov", False)
            mat = sp.dok_matrix(self.dims, dtype=float)                         
            for i in trange(self.basis.num_states,
                            desc="calculate Stark terms", **tqdm_kw):
                # off-diagonal, upper elements only
                for j in range(i + 1, self.basis.num_states):
                    state_1, state_2 = self.basis[i], self.basis[j]
                    st = stark_interaction(state_1, state_2, numerov=numerov)
                    if st is None or st == 0:
                        pass
                    mat[i, j] = st
                    # assume matrix is symmetric
                    mat[j, i] = st
            self._stark_matrix = mat.asformat('csr')
        if dense:
            return Fz * self._stark_matrix.todense()
        return Fz * self._stark_matrix

    def zeeman_matrix(self, Bz=1.0, dense=False, update=False, **kwargs):
        """ Zeeman interaction matrix [atomic units].

        args:
            Bz :: float
                magnetic field [atomic units]
            dense :: Bool
                dense matrix?
            update :: Bool
                recalculate?

        kwrargs:
            tqdm_kw :: dict

        return:
            scipy.sparse.csr_matrix
        """
        if self._zeeman_matrix is None or update:
            tqdm_kw = kwargs.get("tqdm_kw", {})
            mat = sp.dok_matrix(self.dims, dtype=float)
            for i in trange(self.basis.num_states,
                            desc="calculate Zeeman terms", **tqdm_kw):
                # upper elements only
                for j in range(i, self.basis.num_states):
                    state_1, state_2 = self.basis[i], self.basis[j]
                    zm = zeeman_interaction(state_1, state_2)
                    if zm is None or zm == 0:
                        pass
                    mat[i, j] = zm
                    # assume matrix is symmetric
                    if i != j:
                        mat[j, i] = zm
            self._zeeman_matrix = mat.asformat('csr')
        if dense:
            return Bz * self._zeeman_matrix.todense()
        return Bz * self._zeeman_matrix

    def matrix(self, Fz=None, Bz=None, dense=False, **kwargs):
        """ total Hamiltonian matrix [atomic units].

        args:
            Fz  :: float     [atomic units]
            Bz  :: float     [atomic units]

        kwargs:
            dense=False

        return:
            scipy.sparse.csr_matrix
        """
        mat = self.h0_matrix(**kwargs)
        if Fz is not None:
            mat += self.stark_matrix(Fz, **kwargs)
        if Bz is not None:
            mat += self.zeeman_matrix(Bz, **kwargs)
        if dense:
            return mat.todense()
        return mat

    @atomic_units("energy")
    def eigenvectors(self, electric_field=None, magnetic_field=None, units=None, **kwargs):
        """ eigenvalues and eigenvectors of the total Hamiltonian.

        args:
            electric_field  :: float     [V / m]
            magnetic_field  :: float     [T]

        kwargs:
            units="atomic_units"

        return:
            eigenvalues :: np.ndarray, eigenvectors :: np.ndarray
        """
        Fz = electric_field * e * a0 / En_h if electric_field is not None else None
        Bz = magnetic_field * mu_B / En_h if magnetic_field is not None else None
        mat = self.matrix(Fz, Bz, **kwargs).toarray()
        return np.linalg.eigh(mat)

    @atomic_units("energy")
    def eigenvalues(self, electric_field=None, magnetic_field=None, units=None, **kwargs):
        """ eigenvalues of the total Hamiltonian.

        args:
            electric_field  :: float     [V / m]
            magnetic_field  :: float     [T]

        kwargs:
            units="atomic_units"

        return:
            eigenvalues :: np.ndarray
        """
        Fz = electric_field * e * a0 / En_h if electric_field is not None else None
        Bz = magnetic_field * mu_B / En_h if magnetic_field is not None else None
        mat = self.matrix(Fz, Bz, **kwargs).toarray()
        return np.linalg.eigvalsh(mat)

    @atomic_units("energy")
    def stark_map(self, electric_field,
                  magnetic_field=None, elements=False, units=None, **kwargs):
        """ The eigenvalues of the Hamiltonian for a range of electric fields.

        If units is not specified, eigenvalues are returned in atomic units.

        args:
            electric_field      :: numpy.ndarray [V / m]
            magnetic_field=None :: Number        [T]
            elements=False
                    :: bool
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
            mat = (base_matrix + Fz * self._stark_matrix).toarray()
            # diagonalise
            if isinstance(elements, Iterable):
                values[i], vec = np.linalg.eigh(mat)
                amplitudes[i] = np.sum(vec[elements]**2.0, axis=0)
            elif elements:
                values[i], vectors[i] = np.linalg.eigh(mat)
            else:
                values[i] = np.linalg.eigvalsh(mat)
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
            magnetic_field      :: numpy.ndarray    [T]
            electric_field=None :: Number           [V / m]
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
            mat = (base_matrix + Bz * self._zeeman_matrix).toarray()
            # diagonalise
            if isinstance(elements, Iterable):
                values[i], vec = np.linalg.eigh(mat)
                amplitudes[i] = np.sum(vec[elements]**2.0, axis=0)
            elif elements:
                values[i], vectors[i] = np.linalg.eigh(mat)
            else:
                values[i] = np.linalg.eigvalsh(mat)
        # output
        if isinstance(elements, Iterable):
            return values, amplitudes
        elif elements:
            return values, vectors
        else:
            return values

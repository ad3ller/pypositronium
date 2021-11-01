"""Hamiltonian for positronium in electric and magnetic fields."""

from collections.abc import Iterable
import numpy as np
from scipy import linalg
import scipy.sparse as sp
from tqdm import trange
from .energy import energy
from .stark import stark_interaction
from .zeeman import zeeman_interaction
from .constants import atomic_units, mu_B, En_h, e, a0


class Hamiltonian(object):
    """Hamiltonian

    Attributes
    ----------
    basis : Basis
        The basis set of instances of State
    dims : (int, int)
        The dimensions of the Hamiltonian matrix
    e0 : numpy.array
        Field-free energies of the basis states.

    Methods
    -------
    reset()
        Wipe the in-memory cache of sparse matrices.
    e0_matrix()
        Field-free Hamiltonian matrix.
    stark_matrix(Fz)
        Stark interaction matrix.
    zeeman_matrix(Bz)
        Zeeman interaction matrix.
    matrix(Fz, Bz)
        Total Hamiltonian matrix.
    eigenvalues(electric_field, magnetic_field)
        Eigenvalues of the total Hamiltonian.
    eigenvectors(electric_field, magnetic_field)
        Eigenvalues and vectors of the total Hamiltonian.
    stark_map((electric_fields)
        Eigenvalues of the Hamiltonian for a range of electric fields.
    zeeman_map((magnetic_fields)
        Eigenvalues of the Hamiltonian for a range of magnetic fields.

    """

    def __init__(self, basis, sparse_format="csr"):
        """Initialize Hamiltonian.

        Parameters
        ----------
        basis : Basis
            list of State instances.
        sparse_format="csr" : str
            sparse matrix format, e.g., "csr",  "csc" or "array".

        """
        self.basis = basis
        self.dims = (self.basis.num_states, self.basis.num_states)
        self.sparse_format = sparse_format
        self.reset()

    def reset(self):
        """Wipe the in-memory cache of sparse matrices."""
        self._e0_matrix = None
        self._stark_z_matrix = None
        self._zeeman_matrix = None

    @property
    def e0(self):
        """Field-free energy of the basis set [atomic units].

        Returns
        -------
        numpy.ndarray

        """
        return np.array([energy(x) for x in self.basis])

    def e0_matrix(self):
        """Field-free Hamiltonian matrix.

        Returns
        -------
        scipy.sparse.csr_matrix [atomic units]

        """
        if self._e0_matrix is None:
            self._e0_matrix = sp.dia_matrix(
                (self.e0, 0), shape=self.dims, dtype=float
            ).asformat(self.sparse_format)
        return self._e0_matrix

    def stark_matrix(self, **kwargs):
        """Stark interaction matrix.

        Parameters
        ----------
        tqdm_kw={} : dict
            progress bar options, e.g, tqdm_kw={disable : True}
        numerov=False : bool
            use numerov method?

        Returns
        -------
        scipy.sparse.csr_matrix [atomic units]

        """
        if self._stark_z_matrix is None:
            tqdm_kw = kwargs.get("tqdm_kw", {})
            numerov = kwargs.get("numerov", False)
            desc = "calculate Stark terms"
            if numerov:
                desc += " (numerov)"
            mat = sp.dok_matrix(self.dims, dtype=float)
            for i in trange(self.basis.num_states, desc=desc, **tqdm_kw):
                # off-diagonal, upper elements only
                for j in range(i + 1, self.basis.num_states):
                    state_1, state_2 = self.basis[i], self.basis[j]
                    si = stark_interaction(state_1, state_2, numerov=numerov)
                    mat[i, j] = si
                    # assume matrix is symmetric
                    mat[j, i] = si
            self._stark_z_matrix = mat.asformat(self.sparse_format)
        return self._stark_z_matrix

    def zeeman_matrix(self, **kwargs):
        """Zeeman interaction matrix.

        Parameters
        ----------
        tqdm_kw={} : dict
            progress bar options, e.g, tqdm_kw={disable : True}

        Returns
        -------
        scipy.sparse.csr_matrix [atomic units]

        """
        if self._zeeman_matrix is None:
            tqdm_kw = kwargs.get("tqdm_kw", {})
            mat = sp.dok_matrix(self.dims, dtype=float)
            for i in trange(
                self.basis.num_states, desc="calculate Zeeman terms", **tqdm_kw
            ):
                # upper elements only
                for j in range(i, self.basis.num_states):
                    state_1, state_2 = self.basis[i], self.basis[j]
                    zi = zeeman_interaction(state_1, state_2)
                    mat[i, j] = zi
                    # assume matrix is symmetric
                    if i != j:
                        mat[j, i] = zi
            self._zeeman_matrix = mat.asformat(self.sparse_format)
        return self._zeeman_matrix

    def matrix(self, Fz=None, Bz=None, **kwargs):
        """Total Hamiltonian matrix.

        Parameters
        ----------
        Fz : float
            electric field along z [atomic units]
        Bz : float
            magnetic field along z [atomic units]

        Returns
        -------
        scipy.sparse.csr_matrix [atomic units]

        """
        mat = self.e0_matrix()
        if Fz is not None:
            mat += Fz * self.stark_matrix(**kwargs)
        if Bz is not None:
            mat += Bz * self.zeeman_matrix(**kwargs)
        return mat

    @atomic_units("energy")
    def eigenvalues(self, electric_field=0.0, magnetic_field=0.0, units=None, **kwargs):
        """Eigenvalues of the total Hamiltonian.

        Parameters
        ----------
        electric_field : float [V / m]
        magnetic_field : float [T]
        units='atomic_units' : str

        Returns
        -------
        eigenvalues : numpy.ndarray

        """
        Fz = electric_field * e * a0 / En_h
        Bz = magnetic_field * mu_B / En_h
        mat = self.matrix(Fz, Bz, **kwargs)
        if sp.issparse(mat):
            mat = mat.toarray()
        return linalg.eigvalsh(mat)

    @atomic_units("energy")
    def eigenvectors(
        self, electric_field=0.0, magnetic_field=0.0, units=None, **kwargs
    ):
        """Eigenvalues and vectors of the total Hamiltonian.

        Parameters
        ----------
        electric_field : float [V / m]
        magnetic_field : float [T]
        units='atomic_units' : str

        Returns
        -------
        (eigenvalues : numpy.ndarray, eigenvectors : numpy.ndarray)

        """
        Fz = electric_field * e * a0 / En_h
        Bz = magnetic_field * mu_B / En_h
        mat = self.matrix(Fz, Bz, **kwargs)
        if sp.issparse(mat):
            mat = mat.toarray()
        return linalg.eigh(mat)

    @atomic_units("energy")
    def stark_map(
        self, electric_field, magnetic_field=None, elements=False, units=None, **kwargs
    ):
        """Eigenvalues of the Hamiltonian for a range of electric fields.

        Parameters
        ----------
        electric_field : numpy.ndarray [V / m]
        magnetic_field=None : float [T]
        elements=False : bool or Iterable(int)
            calculate the eigenvectors or the sum of the square of
            the specified elements of the eigenvectors?
        units='atomic_units' : str
        tqdm_kw={} : dict
            progress bar options, e.g, tqdm_kw={disable : True}

        Returns
        --------
        if elements==False:
            eigenvalues

        if elements==True:
            eigenvalues, eigenvectors

        if isinstance(elements, Iterable):
            eigenvalues, amplitudes (i.e, the sum of the square of the
                                     specified elements of the eigenvectors)

        Notes
        -----
        A large map with eigenvectors can take up a LOT of memory.

        """
        tqdm_kw = kwargs.get("tqdm_kw", {})
        # initialise output arrays
        num_fields = len(electric_field)
        values = np.empty((num_fields, self.basis.num_states), dtype=float)
        if isinstance(elements, Iterable):
            amplitudes = np.empty((num_fields, self.basis.num_states), dtype=float)
        elif elements:
            vectors = np.empty(
                (num_fields, self.basis.num_states, self.basis.num_states), dtype=float
            )
        # field-free matrix
        base_matrix = self.e0_matrix()
        # magnetic_field
        if magnetic_field is not None and magnetic_field != 0.0:
            Bz = magnetic_field * mu_B / En_h
            base_matrix += Bz * self.zeeman_matrix(**kwargs)
        # get stark matrix
        if self._stark_z_matrix is None:
            self.stark_matrix(**kwargs)
        # loop over electric field values
        for i in trange(num_fields, desc="diagonalise matrix", **tqdm_kw):
            Fz = electric_field[i] * e * a0 / En_h
            mat = base_matrix + Fz * self._stark_z_matrix
            if sp.issparse(mat):
                mat = mat.toarray()
            # diagonalise
            if isinstance(elements, Iterable):
                values[i], vec = linalg.eigh(mat)
                amplitudes[i] = np.sum(vec[elements, :] ** 2.0, axis=0)
            elif elements:
                values[i], vectors[i, :, :] = linalg.eigh(mat)
            else:
                values[i] = linalg.eigvalsh(mat)
        # output
        if isinstance(elements, Iterable):
            return values, amplitudes
        elif elements:
            return values, vectors
        else:
            return values

    @atomic_units("energy")
    def zeeman_map(
        self, magnetic_field, electric_field=None, elements=False, units=None, **kwargs
    ):
        """Eigenvalues of the Hamiltonian for a range of magnetic fields.

        Parameters
        ----------
        magnetic_field : numpy.ndarray [T]
        electric_field=None : float [V/m]
        elements=False : bool or Iterable(int)
            calculate the eigenvectors or the sum of the square of
            the specified elements of the eigenvectors?
        units='atomic_units' : str
        tqdm_kw={} : dict
            progress bar options, e.g, tqdm_kw={disable : True}

        Returns
        --------
        if elements==False:
            eigenvalues

        if elements==True:
            eigenvalues, eigenvectors

        if isinstance(elements, Iterable):
            eigenvalues, amplitudes (i.e, the sum of the square of the
                                    specified elements of the eigenvectors)

        Notes
        -----
        A large map with eigenvectors can take up a LOT of memory.

        """
        tqdm_kw = kwargs.get("tqdm_kw", {})
        # initialise output arrays
        num_fields = len(magnetic_field)
        values = np.empty((num_fields, self.basis.num_states), dtype=float)
        if isinstance(elements, Iterable):
            amplitudes = np.empty((num_fields, self.basis.num_states), dtype=float)
        elif elements:
            vectors = np.empty(
                (num_fields, self.basis.num_states, self.basis.num_states), dtype=float
            )
        # field-free matrix
        base_matrix = self.e0_matrix()
        # electric_field
        if electric_field is not None and electric_field != 0.0:
            Fz = electric_field * e * a0 / En_h
            base_matrix += Fz * self.stark_matrix(**kwargs)
        # get zeeman matrix
        if self._zeeman_matrix is None:
            self.zeeman_matrix(**kwargs)
        # loop over magnetic field values
        for i in trange(num_fields, desc="diagonalise matrix", **tqdm_kw):
            Bz = magnetic_field[i] * mu_B / En_h
            mat = base_matrix + Bz * self._zeeman_matrix
            if sp.issparse(mat):
                mat = mat.toarray()
            # diagonalise
            if isinstance(elements, Iterable):
                values[i], vec = linalg.eigh(mat)
                amplitudes[i] = np.sum(vec[elements, :] ** 2.0, axis=0)
            elif elements:
                values[i], vectors[i, :, :] = linalg.eigh(mat)
            else:
                values[i] = linalg.eigvalsh(mat)
        # output
        if isinstance(elements, Iterable):
            return values, amplitudes
        elif elements:
            return values, vectors
        else:
            return values

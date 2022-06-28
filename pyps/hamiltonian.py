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


@atomic_units("energy")
def eigenenergies(M, units=None, **kwargs):
    """Eigenvalues of M.

    Parameters
    ----------
    M : array or sparse matrix
    units : str, default=None

    Returns
    -------
    numpy.ndarray

    """
    if sp.issparse(M):
        M = M.toarray()
    return linalg.eigvalsh(M, **kwargs)


@atomic_units("energy")
def eigenstates(M, units=None, **kwargs):
    """Eigenvalues and vectors of M.

    Parameters
    ----------
    M : array or sparse matrix
    units : str, default=None

    Returns
    -------
    numpy.ndarray

    """
    if sp.issparse(M):
        M = M.toarray()
    return linalg.eigh(M, **kwargs)


class Hamiltonian(object):
    """Hamiltonian

    Attributes
    ----------
    basis : Basis
        A list of instances of State.
    dims : (int, int)
        dimensions of the Hamiltonian matrix.
    numerov : bool
        use numerov for stark?
    m_alpha6 : bool
        include O(m alpha^6)terms for S and P states?
    upper : bool
        include upper matrix elements?
    sparse_format : str (default="csr")
        sparse matrix format, e.g., "csr",  "csc" or "array".

    Methods
    -------
    reset()
        wipe the in-memory cache of sparse matrices.
    e0()
        field-free Hamiltonian matrix.
    stark(Fz)
        Stark interaction matrix.
    zeeman(Bz)
        Zeeman interaction matrix.
    total(Fz, Bz)
        total Hamiltonian matrix.
    eigenenergies(electric_field, magnetic_field)
        eigenvalues of the total Hamiltonian.
    eigenstates(electric_field, magnetic_field)
        eigenvalues and vectors of the total Hamiltonian.
    stark_map(electric_fields)
        eigenvalues of the Hamiltonian for a range of electric fields.
    zeeman_map(magnetic_fields)
        eigenvalues of the Hamiltonian for a range of magnetic fields.
    """

    def __init__(
        self, basis, numerov=False, m_alpha6=True, upper=False, sparse_format="csr"
    ):
        """Initialize Hamiltonian.

        Parameters
        ----------
        basis : Basis
            list of State instances.
        numerov : bool (default=False)
            use numerov for stark?
        m_alpha6 : bool (default=True)
            include O(m alpha^6) terms for S and P states?
        upper : bool (default=False)
            include upper matrix elements?  
        sparse_format : str (default="csr")
            sparse matrix format.  E.g., "csr",  "csc" or "array".

        """
        self.basis = basis
        self.dims = (self.basis.num_states, self.basis.num_states)
        self.numerov = numerov
        self.m_alpha6 = m_alpha6
        self.upper = upper
        self.sparse_format = sparse_format
        self.reset()

    def reset(self):
        """Clear the in-memory cache of sparse matrices."""
        self._e0_matrix = None
        self._stark_z_matrix = None
        self._zeeman_matrix = None

    def e0(self):
        """Field-free Hamiltonian matrix.

        Returns
        -------
        scipy.sparse.csr_matrix [atomic units]
        """
        if self._e0_matrix is None:
            self._e0_matrix = sp.dia_matrix(
                ([energy(x, m_alpha6=self.m_alpha6) for x in self.basis], 0),
                shape=self.dims,
                dtype=float,
            ).asformat(self.sparse_format)
        return self._e0_matrix

    def stark(self, Fz, **kwargs):
        """Stark interaction matrix.

        Parameters
        ----------
        Fz : float
            electric field along z [atomic units]
        tqdm_kw={} : dict
            progress bar options, e.g, tqdm_kw={disable : True}

        Returns
        -------
        scipy.sparse.csr_matrix [atomic units]
        """
        if self._stark_z_matrix is None:
            tqdm_kw = kwargs.get("tqdm_kw", {})
            desc = "Stark"
            if self.numerov:
                desc += " (numerov)"
            mat = sp.dok_matrix(self.dims, dtype=float)
            for i in trange(self.basis.num_states, desc=desc, **tqdm_kw):
                for j in range(i + 1, self.basis.num_states):
                    state_1, state_2 = self.basis[i], self.basis[j]
                    si = stark_interaction(state_1, state_2, numerov=self.numerov)
                    # upper
                    if self.upper:
                        mat[i, j] = si
                    # lower
                    mat[j, i] = si
            self._stark_z_matrix = mat.asformat(self.sparse_format)
        return Fz * self._stark_z_matrix

    def zeeman(self, Bz, **kwargs):
        """Zeeman interaction matrix.

        Parameters
        ----------
        Bz : float
            magnetic field along z [atomic units]
        tqdm_kw={} : dict
            progress bar options, e.g, tqdm_kw={disable : True}

        Returns
        -------
        scipy.sparse.csr_matrix [atomic units]
        """
        if self._zeeman_matrix is None:
            tqdm_kw = kwargs.get("tqdm_kw", {})
            mat = sp.dok_matrix(self.dims, dtype=float)
            for i in trange(self.basis.num_states, desc="Zeeman", **tqdm_kw):
                for j in range(i, self.basis.num_states):
                    state_1, state_2 = self.basis[i], self.basis[j]
                    zi = zeeman_interaction(state_1, state_2)
                    if i == j:
                        mat[i, j] = zi
                    else:
                        # upper
                        if self.upper:
                            mat[i, j] = zi
                        # lower
                        mat[j, i] = zi
            self._zeeman_matrix = mat.asformat(self.sparse_format)
        return Bz * self._zeeman_matrix

    def total(self, Fz=None, Bz=None, **kwargs):
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
        M = self.e0().copy()
        if Fz is not None and Fz != 0.0:
            M += self.stark(Fz, **kwargs)
        if Bz is not None and Bz != 0.0:
            M += self.zeeman(Bz, **kwargs)
        return M

    @atomic_units("energy")
    def eigenenergies(
        self, electric_field=0.0, magnetic_field=0.0, units=None, **kwargs
    ):
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
        M = self.total(Fz, Bz, **kwargs)
        return eigenenergies(M)

    @atomic_units("energy")
    def eigenstates(self, electric_field=0.0, magnetic_field=0.0, units=None, **kwargs):
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
        M = self.total(Fz, Bz, **kwargs)
        return eigenstates(M)

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
        base_matrix = self.e0().copy()
        # magnetic_field
        if magnetic_field is not None and magnetic_field != 0.0:
            Bz = magnetic_field * mu_B / En_h
            base_matrix += self.zeeman(Bz, **kwargs)
        # get stark matrix
        if self._stark_z_matrix is None:
            self.stark(1.0, **kwargs)
        # loop over electric field values
        for i in trange(num_fields, desc="eigenenergies", **tqdm_kw):
            Fz = electric_field[i] * e * a0 / En_h
            M = base_matrix + Fz * self._stark_z_matrix
            if isinstance(elements, Iterable):
                values[i], vec = eigenstates(M)
                amplitudes[i] = np.sum(vec[elements, :] ** 2.0, axis=0)
            elif elements:
                values[i], vectors[i, :, :] = eigenstates(M)
            else:
                values[i] = eigenenergies(M)
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
        base_matrix = self.e0().copy()
        # electric_field
        if electric_field is not None and electric_field != 0.0:
            Fz = electric_field * e * a0 / En_h
            base_matrix += self.stark(Fz, **kwargs)
        # get zeeman matrix
        if self._zeeman_matrix is None:
            self.zeeman(1.0, **kwargs)
        # loop over magnetic field values
        for i in trange(num_fields, desc="eigenenergies", **tqdm_kw):
            Bz = magnetic_field[i] * mu_B / En_h
            M = base_matrix + Bz * self._zeeman_matrix
            if isinstance(elements, Iterable):
                values[i], vec = eigenstates(M)
                amplitudes[i] = np.sum(vec[elements, :] ** 2.0, axis=0)
            elif elements:
                values[i], vectors[i, :, :] = eigenstates(M)
            else:
                values[i] = eigenenergies(M)
        # output
        if isinstance(elements, Iterable):
            return values, amplitudes
        elif elements:
            return values, vectors
        else:
            return values

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:22:19 2017

@author: Adam

Nb.  To pass kwargs to tqdm progress bar, prepend name using 'tqdm_', e.g., tqdm_disable=True
"""
from operator import attrgetter
import attr
import numpy as np
from tqdm import trange
from sympy.physics.wigner import wigner_3j, wigner_6j
from .numerov import rad_overlap

#CODATA 2014, DOI: 10.1103/RevModPhys.88.035009
c = 299792458.0 ## speed of light in vacuum
h = 6.626070040e-34
hbar = 1.054571800e-34
Ry = 10973731.568508
e = 1.6021766208e-19
m_e = 9.10938356e-31
alpha = 7.2973525664e-3
m_u = 1.660539040e-27
En_h = alpha**2.0 * m_e * c**2.0
a_0 = hbar/ (m_e * c * alpha)
mu_B = e * hbar / (2.0 * m_e)

# positronium
mass_ps = 2.0 * m_e
## reduced electron mass/ m_e
mu_me = 0.5
## Rydberg constant for positronium
Ry_ps = Ry * mu_me
## Bohr radius for Ps
a_ps = a_0 / mu_me

@attr.s()
class State(object):
    """ attrs class to represent the quantum state |n l S J MJ>.
    """
    n = attr.ib(convert=int)
    @n.validator
    def check_n(self, attribute, value):
        if not value > 0:
            raise ValueError("n must be a positive integer.")
    l = attr.ib(convert=int)
    @l.validator
    def check_l(self, attribute, value):
        if not value < self.n:
            raise ValueError("l must be an integer less than n.")
    S = attr.ib(convert=int)
    @S.validator
    def check_S(self, attribute, value):
        if value not in [0, 1]:
            raise ValueError("S must be 0 or 1.")
    J = attr.ib(convert=int)
    @J.validator
    def check_J(self, attribute, value):
        if self.l == 0 and not value == self.S:
            raise ValueError("If l == 0, J must be equal to S.")
        elif self.S == 0 and not value == self.l:
            raise ValueError("If S == 0, J must be equal to l.")
        elif (not self.l - self.S <= value <= self.l + self.S):
            raise ValueError("J must be in range l - S <= J <= l + S.")
    MJ = attr.ib(convert=int)
    @MJ.validator
    def check_MJ(self, attribute, value):
        if (not -self.J <= value <= self.J):
            raise ValueError("MJ must be in the range of -J <= MJ <= J.")

    def __attrs_post_init__(self):
        """ calculate the zero-field energy of the state.
        """
        self.E0 = energy(self.n, self.l, self.S, self.J)
        ## no quantum defect
        self.n_eff = self.n #np.sqrt(- mu_me * 0.5 / self.E0)

    def asdict(self):
        """ quantum numbers as a dictionary.
        """
        return attr.asdict(self)

    def tex(self, show_MJ=True):
        """ Tex string of the form n^{2S + 1}L_{J} (M_J = {MJ})
        """
        L = 'SPDFGHIKLMNOQRTUVWXYZ'[int(self.l%22)]
        tex_str = r'$%d^{%d}'%(self.n, 2*self.S + 1) + L + r'_{%d}'%(self.J)
        if show_MJ:
            tex_str = tex_str + '\,' + r'(M_J = %d)$'%self.MJ
        else:
            tex_str = tex_str + r'$'
        return tex_str

def epsilon(l, S, J):
    """ scaling of the fine structure shift.
    """
    if S == 0:
        # singlet
        epsilon = 0.0
    elif S == 1:
        # triplet
        delta = int(l == 0)
        if J == l + 1:
            omega = (3*l + 4)/ ((l + 1) * (2*l + 3))
        elif J == l:
            omega =  -1.0 / (l*(l + 1))
        elif J == l - 1:
            omega = - (3*l - 1.0)/ (l*(2*l - 1))
        else:
            raise ValueError("The total angular momentum quantum number 'J' must " + \
                             "be in the range l - 1 < J < l + 1")
        epsilon = 7.0 / 6.0 * delta + (1 - delta) / (2.0 * (2 * l + 1)) * omega
    else:
        raise ValueError("The total spin quantum number 'S' must be 0 or 1.")
    return epsilon

def energy_fs(n, l, S, J):
    """ first-order fine structure shift for state |n l S J >
    
        H. A. Bethe and E. E. Salpeter (1957)
        Quantum Mechanics of One- and Two-Electron Systems
    """
    # TODO - special case for l=0 and l=1.
    en = (11.0/ (32 * n**4) + (epsilon(l, S, J) - 1.0 / (2*l + 1)) * 1.0/ (n**3.0)) * alpha**2.0
    return mu_me * en

def energy(n, l, S, J):
    """ energy levels, including fine structure.
    """
    en = - mu_me * 0.5 / (n**2.0) + energy_fs(n, l, S, J)
    return en

class Hamiltonian(object):
    """ The total Hamiltonian matrix.  Each element of the basis set is an
        instance of the class 'State', which represents |n l S J MJ>.
    """
    def __init__(self, n_min, n_max, l_max=None, S=None, MJ=None, MJ_max=None):
        self.n_min = n_min
        self.n_max = n_max
        self.basis = basis_states(n_min, n_max, l_max=l_max, S=S, MJ=MJ, MJ_max=MJ_max)
        self.sort_basis('E0', inplace=True)
        self.num_states = len(self.basis)
        self._h0_matrix = None
        self._stark_matrix = None
        self._zeeman_matrix = None
      
    def sort_basis(self, attribute, inplace=False):
        """ Sort basis on attribute.
        """
        sorted_basis = sorted(self.basis, key=attrgetter(attribute))
        if inplace:
            self.basis = sorted_basis
        return sorted_basis

    def attrib(self, attribute):
        """ List of given attribute values from all elements in the basis, e.g., J or E0.
        """
        return [getattr(el, attribute) for el in self.basis]

    def where(self, attribute, value):
        """ Indexes of where basis.attribute == value.
        """
        arr = self.attrib(attribute)
        return [i for i, x in enumerate(arr) if x == value]

    def h0_matrix(self, cache=True):
        """ Unperturbed Hamiltonian.
        """
        if self._h0_matrix is None or cache is False:
            self._h0_matrix = np.diag(self.attrib('E0'))
        return self._h0_matrix

    def stark_matrix(self, cache=True, **kwargs):
        """ Stark interaction matrix.
        """
        tqdm_kwargs = dict([(x.replace('tqdm_', ''), kwargs[x]) for x in kwargs.keys() if 'tqdm_' in x])
        if self._stark_matrix is None or cache is False:
            self._stark_matrix = np.zeros([self.num_states, self.num_states])
            for i in trange(self.num_states, desc="calculate Stark terms", **tqdm_kwargs):
                # off-diagonal elements only
                for j in range(i + 1, self.num_states):
                    self._stark_matrix[i][j] = stark_int(self.basis[i], self.basis[j])
                    # assume matrix is symmetric
                    self._stark_matrix[j][i] = self._stark_matrix[i][j]
        return self._stark_matrix

    def zeeman_matrix(self, cache=True, **kwargs):
        """ Zeeman interaction matrix.
        """
        tqdm_kwargs = dict([(x.replace('tqdm_', ''), kwargs[x]) for x in kwargs.keys() if 'tqdm_' in x])
        if self._zeeman_matrix is None or cache is False:
            self._zeeman_matrix = np.zeros([self.num_states, self.num_states])
            for i in trange(self.num_states, desc="calculate Zeeman terms", **tqdm_kwargs):
                for j in range(i, self.num_states):
                    self._zeeman_matrix[i][j] = zeeman_int(self.basis[i], self.basis[j])
                    # assume matrix is symmetric
                    if i != j:
                        self._zeeman_matrix[j][i] = self._zeeman_matrix[i][j]
        return self._zeeman_matrix

    def stark_zeeman(self, Efield, Bfield=0.0, **kwargs):
        """ Diagonalise the total Hamiltonian, H_0 + H_S + H_Z, for parallel 
            electric and magnetic fields.
        
            args:
                Efield           dtype: float     units: V / m      

                Bfield=0.0       dtype: float     units: T
            
            kwargs:
                eig_vec=False    dtype: bool

                                 returns the eigenvalues and eigenvectors.

                eig_amp=None     dtype: list

                                 calculate the sum of the square of the amplitudes
                                 of the components of the listed basis states for 
                                 each eigenvector, e.g., eig_amp=[1, 3, 5].
                                 Requires eig_vec=False.
            
        """
        get_eig_vec = kwargs.get('eig_vec', False)
        eig_elements = kwargs.get('eig_amp', None)
        # magnetic field
        if Bfield != 0.0:
            Bz = mu_B * Bfield / En_h
            mat_z =  self.zeeman_matrix(**kwargs)
            H_Z = Bz * mat_z
        else:
            H_Z = 0.0
        # electric field
        if Efield != 0.0:
            Fz = Efield * e * a_0 / En_h
            mat_s = self.stark_matrix(**kwargs)
            H_S = Fz * mat_s / mu_me
        else:
            H_S = 0.0
        # interaction Hamiltonian
        H_int = H_S + H_Z          
        # diagonalise H_tot, assuming matrix is Hermitian.
        if get_eig_vec:
            # eigenvalues and eigenvectors
            eig_val, eig_vec = np.linalg.eigh(self.h0_matrix() + H_int)
            return eig_val * En_h, eig_vec
        elif eig_elements is not None:
            # eigenvalues and partial eigenvector amplitudes
            eig_val, vec = np.linalg.eigh(self.h0_matrix() + H_int)
            eig_amp = np.sum(vec[eig_elements]**2.0, axis=0)
            return eig_val, eig_amp
        else:
            # eigenvalues
            eig_val = np.linalg.eigh(self.h0_matrix() + H_int)[0]
            return eig_val * En_h

        
    def stark_map(self, Efield, Bfield=0.0, **kwargs):
        """ The eigenvalues of H_0 + H_S + H_Z, for a range of electric fields.
        
            args:
                Efield           dtype: list      units: V / m      

                Bfield=0.0       dtype: float     units: T
            
            kwargs:
                eig_vec=False    dtype: bool

                                 returns the eigenvalues and eigenvectors for 
                                 every field value.

                eig_amp=None     dtype: list

                                 calculate the sum of the square of the amplitudes
                                 of the components of the listed basis states for 
                                 each eigenvector, e.g., eig_amp=[1, 3, 5].
                                 Requires eig_vec=False.

            Nb. A large map with eignvectors can take up a LOT of memory.
        """
        tqdm_kwargs = dict([(x.replace('tqdm_', ''), kwargs[x]) for x in kwargs.keys() if 'tqdm_' in x])
        get_eig_vec = kwargs.get('eig_vec', False)
        eig_elements = kwargs.get('eig_amp', None)
        num_fields = len(Efield)
        # initialise output arrays
        eig_val = np.empty((num_fields, self.num_states), dtype=float)
        if get_eig_vec:
            eig_vec = np.empty((num_fields, self.num_states, self.num_states), dtype=float)
        elif eig_elements is not None:
            eig_amp = np.empty((num_fields, self.num_states), dtype=float)
        # optional magentic field
        if Bfield != 0.0:
            Bz = mu_B * Bfield / En_h
            mat_z =  self.zeeman_matrix(**kwargs)
            H_Z = Bz * mat_z
        # loop over electric field values
        mat_s = self.stark_matrix(**kwargs)
        for i in trange(num_fields, desc="diagonalise Hamiltonian", **tqdm_kwargs):
            Fz = Efield[i] * e * a_0 / En_h
            H_S = Fz * mat_s / mu_me
            if Bfield == 0:
                H_int = H_S
            else:
                H_int = H_S + H_Z          
            # diagonalise, assuming matrix is Hermitian.
            if get_eig_vec:
                # eigenvalues and eigenvectors
                eig_val[i], eig_vec[i] = np.linalg.eigh(self.h0_matrix() + H_int)
            elif eig_elements is not None:
                # eigenvalues and partial eigenvector amplitudes
                eig_val[i], vec = np.linalg.eigh(self.h0_matrix() + H_int)
                eig_amp[i] = np.sum(vec[eig_elements]**2.0, axis=0)            
            else:
                # eigenvalues
                eig_val[i] = np.linalg.eigh(self.h0_matrix() + H_int)[0]
        # output
        if get_eig_vec:
            return eig_val * En_h, eig_vec
        elif eig_elements is not None:
            return eig_val * En_h, eig_amp
        else:
            return eig_val * En_h

    def zeeman_map(self, Bfield, Efield=0.0, **kwargs):
        """ The eigenvalues of H_0 + H_S + H_Z, for a range of magnetic fields.
        
            args:
                Bfield           dtype: list      units: T      

                Efield=0.0       dtype: float     units: V / m
            
            kwargs:
                eig_vec=False    dtype: bool

                                 returns the eigenvalues and eigenvectors for 
                                 every field value.

                eig_amp=None     dtype: list

                                 calculate the sum of the square of the amplitudes
                                 of the components of the listed basis states for 
                                 each eigenvector, e.g., eig_amp=[1, 3, 5].
                                 Requires eig_vec=False.
            
            Nb. A large map with eignvectors can take up a LOT of memory.
        """
        tqdm_kwargs = dict([(x.replace('tqdm_', ''), kwargs[x]) for x in kwargs.keys() if 'tqdm_' in x])
        get_eig_vec = kwargs.get('eig_vec', False)
        eig_elements = kwargs.get('eig_amp', None)
        num_fields = len(Bfield)
        # initialise output arrays
        eig_val = np.empty((num_fields, self.num_states), dtype=float)
        if get_eig_vec:
            eig_vec = np.empty((num_fields, self.num_states, self.num_states), dtype=float)
        elif eig_elements is not None:
            eig_amp = np.empty((num_fields, self.num_states), dtype=float)
        # optional electric field
        if Efield != 0.0:
            Fz = Efield * e * a_0 / En_h
            mat_s = self.stark_matrix(**kwargs)
            H_S = Fz * mat_s / mu_me
        # loop over magnetic field values
        mat_z = self.zeeman_matrix(**kwargs)
        for i in trange(num_fields, desc="diagonalise Hamiltonian", **tqdm_kwargs):
            Bz = mu_B * Bfield[i] / En_h
            H_Z =  Bz * mat_z
            if Efield == 0.0:
                H_int = H_Z
            else:
                H_int = H_S + H_Z
            # diagonalise, assuming matrix is Hermitian.
            if get_eig_vec:
                # eigenvalues and eigenvectors
                eig_val[i], eig_vec[i] = np.linalg.eigh(self.h0_matrix() + H_int)
            elif eig_elements is not None:
                # eigenvalues and partial eigenvector amplitudes
                eig_val[i], vec = np.linalg.eigh(self.h0_matrix() + H_int)
                eig_amp[i] = np.sum(vec[eig_elements]**2.0, axis=0)            
            else:
                # eigenvalues
                eig_val[i] = np.linalg.eigh(self.h0_matrix() + H_int)[0]
        # output
        if get_eig_vec:
            return eig_val * En_h, eig_vec
        elif eig_elements is not None:
            return eig_val * En_h, eig_amp
        else:
            return eig_val * En_h

def basis_states(n_min, n_max, **kwargs):
    """ Generate the basis set: a list of instances of the attrs class State that 
        satisfy the given ranges of quantum numbers.  By default, all possible 
        states in the range of n_min to n_max are returned.
        
        args:
            n_min             Minimum value of the principal quantum number.

            n_max             Maximum value of the principal quantum number.
        
        kwargs:
            l_max = None      Maximum value of the orbital angular momentum quantum number.
                              If l_max is None 0 < l < n.

            S = None          Value of the total spin quanum number. If S is None S = [0, 1].

            MJ = None         Value of the projection of the total angular momentum
                              quantum number. If MJ is None -J <= MJ <= J.

            MJ_max = None     Maximum of the absolute value of the projection of the
                              total angular momentum quantum number. If MJ_max and MJ
                              are None -J <= MJ <= J.
    """
    l_max = kwargs.get('l_max', None)
    S = kwargs.get('S', None)
    MJ = kwargs.get('MJ', None)
    MJ_max = kwargs.get('MJ_max', None)
    basis = []
    n_rng = np.arange(n_min, n_max + 1, dtype='int')
    # loop over n range
    for n in n_rng:
        if l_max is not None:
            _l_max = min(l_max, n - 1)
        else:
            _l_max = n - 1
        l_rng = np.arange(0, _l_max + 1, dtype='int')
        # loop over l range
        for l in l_rng:
            if S is None:
                # singlet and triplet states
                S_vals = [0, 1]
            else:
                S_vals = [S]
            for _S in S_vals:
                # find all J vals and MJ substates
                if l == 0:
                    J = _S
                    if MJ is None:
                        for _MJ in np.arange(-J, J + 1):
                            if MJ_max is None or abs(_MJ) <= MJ_max:
                                basis.append(State(n, l, _S, J, _MJ))
                    elif -J <= MJ <= J:
                        basis.append(State(n, l, _S, J, MJ))
                elif _S == 0:
                    J = l
                    if MJ is None:
                        for _MJ in np.arange(-J, J + 1):
                            if MJ_max is None or abs(_MJ) <= MJ_max:
                                basis.append(State(n, l, _S, J, _MJ))
                    elif -J <= MJ <= J:
                        basis.append(State(n, l, _S, J, MJ))
                else:
                    for J in [l + _S, l, l - _S]:
                        if MJ is None:
                            for _MJ in np.arange(-J, J + 1):
                                if MJ_max is None or abs(_MJ) <= MJ_max:
                                    basis.append(State(n, l, _S, J, _MJ))
                        elif -J <= MJ <= J:
                            basis.append(State(n, l, _S, J, MJ))
    return basis

def stark_int(state_1, state_2):
    """ Stark interaction between two states,
    
        <n' l' S' J' MJ'| H_S |n l S J MJ>.
    """
    delta_l = state_2.l - state_1.l
    delta_S = state_2.S - state_1.S
    l_max = max(state_1.l, state_2.l)
    if abs(delta_l) == 1 and delta_S == 0:
        return (-1.0)**(state_1.S +1 + state_2.MJ) * \
                np.sqrt(l_max * (2*state_2.J + 1)*(2*state_1.J + 1)) * \
                wigner_3j(state_2.J, 1, state_1.J, -state_2.MJ, 0, state_1.MJ) * \
                wigner_6j(state_2.S, state_2.l, state_2.J, 1, state_1.J, state_1.l) * \
                rad_overlap(state_1.n_eff, state_1.l, state_2.n_eff, state_2.l)
    else:
        return 0.0

def zeeman_int(state_1, state_2):
    """ Zeeman interaction between two states,
    
        <n' l' S' J' MJ'| H_Zeeman |n l S J MJ>.
    """
    delta_l = state_2.l - state_1.l
    if delta_l == 0:
        return (-1.0)**(state_1.l + state_1.MJ) * \
               ((-1.0)**(state_1.S + state_2.S)- 1.0) * \
               np.sqrt(3.0 * (2*state_2.J + 1) * (2*state_1.J + 1)) * \
               wigner_3j(state_2.J, 1, state_1.J, -state_2.MJ, 0, state_1.MJ) * \
               wigner_6j(state_2.S, state_2.l, state_2.J, state_1.J, 1, state_1.S)
    else:
        return 0.0
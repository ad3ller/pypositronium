positronium fs
==============

v0.0.3

Calculate the energy levels of positronium in parallel electric and magnetic fields.

Fine structure is included to first order using the formula given on page 117 of:

  | Quantum Mechanics of One- And Two-Electron Atoms  
  | by Hans a. Bethe and Edwin E. Salpeter  
  | ISBN 978-1-61427-622-7

The Stark and Zeeman matrices are constructed using the equations given in:

  | A. M. Alonso et al., Phys. Rev. A, 93, 012506 (2016) 
  | https://dx.doi.org/10.1103/PhysRevA.93.012506

Radial wavefunctions are obtained using the Numerov method, as described by:

  | M. L. Zimmerman et al., Phys. Rev. A, 20, 2251 (1979)
  | https://dx.doi.org/10.1103/PhysRevA.20.2251

Install
-------

Using `anaconda <https://anaconda.org/>`_, the main requirements can be installed with conda:

.. code-block:: bash

   conda install scipy sympy numba


Then install using setuptools (this will also install `tqdm <https://github.com/tqdm/tqdm>`_).

.. code-block:: bash

   git clone https://github.com/ad3ller/positronium_fs
   cd ./positronium_fs
   python setup.py install

The package can now be imported into python as *psfs*.

Basic Usage
-----------

.. code:: ipython3

    >>> from psfs import Basis, Hamiltonian
    >>> mat = Hamiltonian(Basis(n_values=range(1, 4)))
    >>> print('number of basis states:', '%d'%mat.basis.num_states)


.. parsed-literal::

    number of basis states: 56
    

.. code:: ipython3

    >>> # e.g., the 10th element of the basis set
    >>> mat.basis[10]


.. parsed-literal::

    State(n=2, L=1, S=0, J=1, MJ=0)

.. code:: ipython3

    >>> # ket notation
    >>> print(mat.basis[10])


.. parsed-literal::

    ❘ 2 1 0 1 0 ⟩

An instance of the `Hamiltonian` class is initialised using a Basis, which is a UserList of instances of the dataclass `State`.
`Hamiltonian` has methods `stark_map()` and `zeeman_map()`, which use the basis set to calculate energy eigenvalues in a range
of electric or magnetic fields.

See the notebooks for examples.

Some of the notebooks require https://github.com/ad3ller/Stark-map-tools.

Version information
-------------------

===================  ====================================================
Python               3.7.3 64bit [GCC 7.3.0]
IPython              7.4.0
OS                   Linux 5.0.0 15 generic x86_64 with debian buster sid
matplotlib           3.0.3
numba                0.43.1
numpy                1.16.2
scipy                1.2.1
sympy                1.3
tabulate             0.8.3
tqdm                 4.31.1
version_information  1.0.3
===================  ====================================================


Examples
--------

This code has not been tested extensively.  But several published calculations have been successfully reproduced.

----

S\. M. Curry, *Phys. Rev. A*, **7** (2), 447 (1973) https://dx.doi.org/10.1103/PhysRevA.7.447

.. figure:: ./images/zeeman_n2.png
   :width: 250px
   
   **Fig. 2** Pure Zeeman effect in the first excited states of positronium.

----

A\. M. Alonso *et al.*, *Phys. Rev. A*, **93**, 012506 (2016) https://dx.doi.org/10.1103/PhysRevA.93.012506
 
.. figure:: ./images/stark_n2.png
   :width: 450px
   
   **Fig. 6 a) & b)** Dependence of the relative energies of all n=2 eigenstates in Ps on electric-field strength (a) in the absence of a magnetic field and (b) in a parallel magnetic field of B=130 G.
   
.. figure:: ./images/stark_n2_zoom.png
   :width: 450px
   
   **Fig. 6 c)** An expanded view of the avoided crossing.

----

G\. Dufour *et al.*, *Adv. High En. Phys.*, **2015**, 379642 (2015) https://dx.doi.org/10.1155/2015/379642

.. figure:: ./images/stark_n31_singlet_MJ2_MJ29.png
   :width: 450px

   **Fig. 11**: Stark states of n=30 and 31 states of Ps, with m=2 (grey dashed) and m=29 (black). In the n=30 level, the m=29 state is a circular state and experiences no first-order Stark shift and only a very weak second-order shift, as explained in the text.

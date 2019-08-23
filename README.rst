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

The main requirements can be installed with `conda <https://anaconda.org/>`_:

.. code-block:: bash

   conda install numpy scipy sympy tqdm


Next, build and install `numerov <https://github.com/ad3ller/numerov>`_.

And finally, clone the souce code and then install the package using setuptools.

.. code-block:: bash

   git clone https://github.com/ad3ller/positronium_fs
   cd ./positronium_fs
   python setup.py install


Basic Usage
-----------

`Basis` is a list of instances of the dataclass `State`.

.. code:: ipython3

    >>> from psfs import Basis, Hamiltonian
    >>> basis = Basis(n_values=range(1, 4))
    >>> print(f'number of basis states: {basis.num_states}')


.. parsed-literal::

    number of basis states: 56
    

.. code:: ipython3

    >>> # e.g., the 10th element of the basis set
    >>> basis[10]


.. parsed-literal::

    State(n=2, L=1, S=0, J=1, MJ=0)

.. code:: ipython3

    >>> # ket notation
    >>> print(basis[10])


.. parsed-literal::

    ❘ 2 1 0 1 0 ⟩

The `Hamiltonian` class is initialised using a basis.  

.. code:: ipython3

    >>> # initialize
    >>> mat = Hamiltonian(basis)

The method `eigvals()` returns the eigenvalues.

.. code:: ipython3

    >>> electric_field = 10.1   # [V / m]
    >>> magnetic_field = 0.1    # [T]
    >>> en = mat.eigvals(electric_field, magnetic_field, units="eV")
    >>> print(en[:5])

.. parsed-literal::

    [-6.80332213 -6.8024767  -6.8024767  -6.80247654 -1.70078788]

The methods `stark_map()` and `zeeman_map()` calculate the eigenvalues for a range of electric or magnetic fields.

See the notebooks for examples.

Some of the notebooks require https://github.com/ad3ller/Stark-map-tools.

Version information
-------------------

==========  ====================================================
Python      3.7.3 64bit [GCC 7.3.0]
IPython     7.6.1
OS          Linux 5.0.0 23 generic x86_64 with debian buster sid
cython      0.29.12
matplotlib  3.1.0
numerov     0.0.4
numpy       1.16.4
sympy       1.4
tqdm        4.32.1
==========  ====================================================


Examples
--------

This code has *not* been tested extensively.  But several published calculations have been successfully reproduced.

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

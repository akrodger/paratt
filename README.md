# paratt
An MPI Parallel Tensor Train Library for solutions to high dimensional
partial differential equations.

# Dependencies & Building
We require an implementation of MPI, a C compiler, and a Fortran compiler.
We also require BLAS and LAPACK linkable libraries are expected to be
installed. The location of these can be modified in the root directory makefile
by editing the make variables "BLAS" and "LAPACK". LAPACK expected as version
3.9.1 or later, though no testing has been done outside of version 3.9.1.
We have not yet found a BLAS version that does not work, but take caution if
you are using your own custom compiled version. Reference blas which is shipped
with LAPACK 3.9.1 at github repository

https://github.com/Reference-LAPACK/lapack/tree/lapack-3.9.1

has been tested to work.

After specifying your dependences, run ``make'' from the root directory.

# About
This code is an extension and fork of the experimental software called
MPI-ATTAC, which can be found at

https://gitlab.com/aldaas/mpi_attac

Similar to that software, this code is only intended for evaluation by expert
users and is in ongoing development. The code shares various pieces of
infrastructure with MPI-ATTAC, though uses an alternative implementation
of the TSQR algorithm and an alternative criterion for Tensor Train low-rank
truncation. The code also contains a data structure for memory management of
distributed memory parallel tensor trains and a system for writing and reading
their data to and from the file system for data I/O. There is also a library
for use of distributed memory parallel tensor trains in ordinary differential
equations. This code has only been tested on Linux computing systems with
the GNU Compiler Collection and OpenMPI.

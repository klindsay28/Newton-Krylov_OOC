==========
Background
==========

The application that motivated the development of Newton-Krylov_OOC is spinning up
biogeochemical tracers in an ocean general circulation model (OGCM).
The target model configuration is multi-year runs with O(10\ :sup:`6-7`) grid points.
Such runs take hours of time on a supercomputer and are typically done through a batch
submission system.

Literature on the topic demonstrates that Newton-Krylov (NK) based solvers can be used
effectively to solve this problem :cite:`Li_Primeau_OceMod_2008,Khatiwala_OceMod_2008`.
These solvers utilize Newton's method to solve for an initial model state such that end
state from running the model forward in time is the same as the initial state.
Newton's method is a general iterative method for solving non-linear equations.
Successive iterations are computed by adding an increment to the previous iterate.
The increment is the solution of a large system of linear equations.
It turns out that it is not practical to directly solve the system of linear equations
that determine the increment, so an iterative solver is used.
Krylov-based iterative solvers are well suited to solve the system of equations that
arise.
Each iteration of the Krylov solver requires a forward model run, which in our target
application is a multi-year run.
Overall, using an NK solver requires at least 10s of forward model runs.

An excellent book on NK solvers, :cite:`Kelley_Newtons_Method`, advises that it is
preferable to use an existing Newton–Krylov implementation instead of implementing your
own.
However, existing implementations of NK solvers have some features that make their usage
impractical for the target model configuration.

#. In-Core Memory Requirement:
   Existing NK solvers utilize a driver program that stays resident in memory
   while function evaluations occur.
   On some computing systems, it is not possible to have the driver program reside in
   memory for the duration of multiple model runs.


#. Solving One Problem at a Time:

.. bibliography:: ../references.bib

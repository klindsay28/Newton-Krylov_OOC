.. _terminology:

===========
Terminology
===========

In Newton-Krylov_OOC, Newton's method is being applied to solve a system of non-linear equations:

.. math::
   F(X) = 0

In this solver, we consider :math:`X` to represent a state vector from a model, and :math:`F` is a function that maps from one state vector to another.
In the application to spinning up tracers in an OGCM, the model is the OGCM, the state :math:`X` is the vector of tracer values being spun up, and the function :math:`F` is the change in state that occurs when the model is run forward from state :math:`X`.

.. _terminology_tracers:

The state :math:`X` is treated as a collection of tracer values.
This arises in the OGCM application from having multiple tracer values on the same model grid.
Another example is water and carbon content in layers of soil in a land model.
In practice, some tracers are decoupled from other tracers.
For example, CFC tracers in an OGCM are independent of noble gas tracers.
We refer to collections of related or coupled tracers as a tracer module.
The solver solves :math:`F` for each tracer module independently.

.. _terminology_regions:

The solver also has a concept of regions, that tracer values in subsets of the state space are decoupled from other subsets of the state space.
An example of this in the OGCM application is that tracer values in the Black Sea are decoupled from tracer values in the open ocean, if the grid does not resolve exchange through the Bosporus.
In the land model application, different columns in a land model are typically independent of each other.
The solver solves :math:`F` for each tracer module in each separate region independently.

Summarizing, the solver is applied to solve a function that operates on a model state; the model state is comprised of a collection of tracer modules, each of which is comprised of related or coupled tracers.
Additionally, tracer values in different regions are treated independently of each other.

We note that this terminology of tracers, tracer modules, and regions is based primarily on the application of the solver to spinning up tracers in an OGCM.
The mathematical statement that tracer values from different modules or regions are independent is that :math:`\partial F/\partial X` has a block structure corresponding to tracer modules and regions, and that off-diagonal blocks in :math:`\partial F/\partial X` are zero.

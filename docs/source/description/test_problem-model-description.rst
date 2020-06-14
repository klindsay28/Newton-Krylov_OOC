.. _test_problem-model-description:

==============================
test_problem Model Description
==============================

The test_problem model is a time-varying one-dimensional column model.
The model has multiple tracer modules that have one or more related tracers.
Tracer tendencies are determined by vertical mixing and module specific source/sink terms.
The model evolves the tracers forward in time with these tendencies for one 365-day year.
The function :math:`F(X)` that the Newton-Krylov (NK) solver is finding a root of is the change in tracer that occurs when the model is run forward from state :math:`X`.
That is, the solver solves for an initial tracer state such that end state from running the model forward in time is the same as the initial state.

Vertical mixing coefficients are large, 1 m\ :sup:`2` s\ :sup:`-1`, in a boundary layer region, and small, 10\ :sup:`-5` m\ :sup:`2` s\ :sup:`-1` below the boundary layer, with a transition across the boundary layer depth.
The boundary layer depth has a sinusoidal dependence on time.

The implementation of the test_problem model in the solver includes supports the tracer modules ``iage``, ``phosphorus``, and ``dye_decay_{suff}``.
Information on adding support for other tracer modules is included in the :ref:`developer's guide <add-tracer-module-test_problem>`.

The ``iage`` tracer module has a single tracer, ideal age, which has a source term of 1 year/year.
A surface flux nudging the surface value to 0 with a piston velocity of 240 m day\ :sup:`-1` is applied to keep the surface value close to 0.
This is equivalent to restoring to 0 over the top 10 m with a rate of 1 hr\ :sup:`-1`.

The ``phosphorus`` tracer module has 3 tracers: phosphate (po4), dissolved organic phosphorus (dop), and particulate organic phosphorus (pop).
There is uptake of po4, representing primary productivity.
Uptake of po4 is the product of an optimal uptake rate, a Michealis-Menten based po4 limiting term, and a light-limitation term.
The light-limitation terms is constant in time and decays exponentially with depth.
Uptake of po4 is routed instantaneously to dop and pop, and these remineralize back to po4.
pop also sinks through the column.
Total phosphorus, po4+dop+pop, in the model is conserved.

The ``dye_decay_{suff}`` tracer module is a :ref:`parameterized tracer module <parameterized-tracer-module>`.
These tracers have a surface flux injecting 1 mol/m\ :sup:`2`/year of tracer content, and they have a decay rate
The ``{suff}`` substring is a 3 character string of digits ``nnn`` prescribing a first-order decay rate of ``nnn/1000``/year.
It is primarily included to demonstrate the usage of parameterized tracer modules.

--------------
Shadow Tracers
--------------

The ``phosphorus`` model utilizes shadow tracers.
These tracers have the some formulation as their real tracer counterparts, except that po4 uptake for the shadow tracers is determined by the real po4 tracer, and the shadow po4 tracer is restored to the real po4 tracer.
This restoring term is subtracted from the shadow dop and pop tracers, in order to conserve total shadow phosphorus.
Newton's method in the NK solver is applied to the shadow tracers and ignores their real tracer counterpaerts.
Because po4 uptake for the shadow tracer is independent of the shadow tracers, it is as if the shadow tracers are being spun up with a fixed productivity field.
The restoring of the po4 shadow tracer to the real po4 tracer keeps the shadow tracers from getting unphysical values.

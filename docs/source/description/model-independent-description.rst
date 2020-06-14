=============================
Model Independent Description
=============================

As described in the :ref:`terminology section<terminology>` of the NK solver description, the NK solver is applied to a model that is comprised of a collections of tracer modules, each of which is comprised of tracers.
This section describes details of using the NK solver that are independent of which model the solver is being applied to.

--------------
Shadow Tracers
--------------

Some models utilize shadow tracers that are formulated to mostly track their real tracer counterparts.
For these models, Newton's method is applied to the shadow tracers and ignores the real tracer counterparts.
The purpose of doing this is to enable the disabling of feedbacks in the tracer evolution equations from Newton's method, feedbacks that might behave non-linearly on timescales much shorter than the model forward integration duration.
An example is spinning up biogeochemical tracers in an OGCM, and utilizing shadow nutrient tracers whose uptake is taken from their real tracer counterparts.
This effectively spin up the shadow nutrients with fixed productivity fields.
The details of the shadowing are confined to the model implementation, and do not propagate into the NK solver.
At the end of each Newton iteration, shadow tracers are copied to the real counterparts.

----------------------
Fixed Point Iterations
----------------------

The Newton solver generates, for each iteration, an increment that satisfies an Armijo residual improvement criteria.
The sum of this increment and the current Newton iterate is a provisional next iterate.
Before proceeding to the next Newton iteration, the solver performs a number of forward model runs that are initialized with this provisional iterate, to allow short time scale adjustments in the model to occur.
Fixed point iterations tend to be effective at spinning up processes whose timescales are much shorter than the model forward integration duration.
These fixed point iterations are performed after copying shadow tracers to their real tracer counterparts, for those models that have shadow tracers.
In this use case, the fixed point iterations enable the shadowed tracers to adjust to the Newton increment.
The number of these fixed point iterations, which defaults to 1, can be modified by changing ``post_newton_fp_iter`` in the cfg file.

------------------
Driver Persistence
------------------

The NK solver can either run with the driver persistent in memory, or it can invoke itself after a forward model run and exit.
The latter approach is necessary when the computing environment does not allow for the driver task to persist in memory for the amount of time that it takes to perform multiple forward model runs.
The cime_pop model is implemented such that the solver exits immediately after submitting the forward model run to a batch job submission system, reducing the amount of time that the solver resides in memory.

==========================
test_problem Model Details
==========================

-----
Usage
-----

Most options for the solver and the test_problem model are in a cfg file.
The default location of the cfg file is ``$TOP/models/test_problem/newton_krylov.cfg``,
where ``$TOP`` is the toplevel directory of the repo.

To set up usage of the test_problem model, run the following command from ``$TOP``
::

  ./models/test_problem/setup_solver.sh

Running ``./models/test_problem/setup_solver.sh -h`` shows what command line options are
available, such as the path of the cfg file.
The ``setup_solver.sh`` script does the following

#. Create a work directory.
   The location of the work directory is in the cfg file.
   The default location is in the users home directory.
   The work directory for test_problem is small.
#. Create a vertical grid file in the work directory.
   The vertical grid can be configured with ``setup_solver.sh`` command line options.
   The default is 30 layers spanning 675 m with a top-layer thickness of 10 m.
   The defaults can be overwritten with arguments to the ``setup_solver.sh`` script.
#. Create an initial model state on the generated vertical grid.
   The initial state is hard-wired to be written to a subdirectory of the work directory
   named ``gen_ic``.
#. Run the model forward from the generated initial state for a number of years.
   The number of years run, which defaults 2, can be modified by with the ``--fp_cnt``
   argument to the ``setup_solver.sh`` script.
   The result of these forward model runs is written to the ``gen_ic`` directory.
#. Invoke ``gen_invoker_script``, to generate the solver invoker script.


Running the invocation script generated in the last step will start the NK solver.
The solver will run until a convergence criteria is met, or the maximum number of Newton
iterations is exceeded.
Both of these options are in the cfg file.
The default settings yield convergence for the ideal age tracer module, but not the
phosphorus tracer module.

By default, the solver solves the test_problem model with the driver persistent in memory.
The test_problem model is small enough that this is feasible.
If the ``reinvoke`` setting in the cfg file is set to True,
then the solver reinvokes itself after each forward model run and exits.
This exercises the the out-of-core functionality that is necessary for when the NK solver
is applied to large models.

-----------
Description
-----------

The test_problem model is a time-varying one-dimensional column model.
The model has multiple tracer modules that have one or more related tracers.
Tracer tendencies are determined by vertical mixing and module specific source/sink terms.
The model evolves the tracers forward in time for one 365-day year.
The Newton-Krylov (NK) solver solves for an initial tracer state such that end state from
running the model forward in time is the same as the initial state.

Vertical mixing coefficients are large, 1 m\ :sup:`2` s\ :sup:`-1`, in a boundary layer
region, and small, 10\ :sup:`-5` m\ :sup:`2` s\ :sup:`-1` below the boundary layer,
with a transition across the boundary layer depth.
The boundary layer depth has a sinusoidal dependence on time.

The model has ideal age and phosphorus tracer modules.
The ideal age tracer has a source term of 1 year/year, and is restored to 0 in the top
layer with a 1-hour restoring timescale.

The phosphorus tracer module has 3 tracers: phosphate (po4), dissolved organic phosphorus
(dop), and particulate organic phosphorus (pop).
There is uptake of po4, representing primary productivity.
Uptake is the product of an optimal uptake rate, a Michealis-Menten based po4 limiting
term, and a light-limitation term.
The light-limitation terms is constant in time and decays exponentially with depth.
Uptake is routed instantaneously to dop and pop, and these remineralize back to po4.
pop also sinks through the column.
Total phosphorus in the model is conserved.

--------------
Shadow Tracers
--------------

The phosphorus model utilizes shadow tracers.
These tracers have the some formulation as their real tracer counterparts,
except that po4 uptake for the shadow tracers is determined by the real po4 tracer,
and the shadow po4 tracer is restored to the real po4 tracer.
This restoring term is subtracted from the shadow dop and pop tracers, in order to
conserve total shadow phosphorus.
Newton's method in the NK solver is applied to the shadow tracers and ignores their
real tracer counterpaerts.
Because po4 uptake for the shadow tracer is independent of the shadow tracers,
it is as if the shadow tracers are being spun up with a fixed productivity field.
The restoring of the po4 shadow tracer to the real po4 tracer keeps the shadow tracers
from getting unphysical values.

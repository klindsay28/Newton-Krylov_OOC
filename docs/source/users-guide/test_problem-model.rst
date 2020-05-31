==========================
test_problem Model Details
==========================

-----------
Description
-----------

The test_problem model is a time-varying one-dimensional column model.
The model has multiple tracer modules that have one or more related tracers.
Tracer tendencies are determined by vertical mixing and module specific source/sink terms.
The model evolves the tracers forward in time with these tendencies for one 365-day year.
The function :math:`F(X)` that the Newton-Krylov (NK) solver is finding a root of is the
change in tracer that occurs when the model is run forward from state :math:`X`.
That is, the solver solves for an initial tracer state such that end state from running
the model forward in time is the same as the initial state.

Vertical mixing coefficients are large, 1 m\ :sup:`2` s\ :sup:`-1`, in a boundary layer
region, and small, 10\ :sup:`-5` m\ :sup:`2` s\ :sup:`-1` below the boundary layer,
with a transition across the boundary layer depth.
The boundary layer depth has a sinusoidal dependence on time.

The model has two tracer modules available: ``iage`` and ``phosphorus``.
Information on adding tracer modules to the test_problem model are included in
the :ref:`developer's guide <add_modules_test_problem>`.

The ``iage`` tracer module has a single tracer, ideal age, which has a source term of 1
year/year.
A surface flux nudging the surface value to 0 with a piston velocity of 240 m day\
:sup:`-1` is applied to keep the surface value close to 0.
This is equivalent to restoring to 0 over the top 10 m with a rate of 1 hr\ :sup:`-1`.

The ``phosphorus`` tracer module has 3 tracers: phosphate (po4), dissolved organic
phosphorus (dop), and particulate organic phosphorus (pop).
There is uptake of po4, representing primary productivity.
Uptake of po4 is the product of an optimal uptake rate, a Michealis-Menten based po4
limiting term, and a light-limitation term.
The light-limitation terms is constant in time and decays exponentially with depth.
Uptake of po4 is routed instantaneously to dop and pop, and these remineralize back to
po4.
pop also sinks through the column.
Total phosphorus, po4+dop+pop, in the model is conserved.

--------------
Shadow Tracers
--------------

The ``phosphorus`` model utilizes shadow tracers.
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

-----
Usage
-----

Most options for the solver and the test_problem model are in the cfg file.
The default location of the cfg file is ``$TOP/input/test_problem/newton_krylov.cfg``,
where ``$TOP`` is the toplevel directory of the repo.
Perform the following steps to spin up tracers in the test_problem model.

~~~~~~
Step 1
~~~~~~

Customize variable settings in the cfg file.
We recommend that the user makes these modifications on a copy of the file from the repo,
to be able to preserve the settings for a particular application of the solver, and to
avoid conflicts if the repo copy is updated.
The following variables are the most commonly set by the user:

* ``workdir``: directory where solver generated files are stored
* ``newton_rel_tol``: relative tolerance for Newton convergence; The solver is considered
  converged if :math:`|F(X)| < \text{newton_rel_tol} \cdot |X|` for each tracer module
  and region.
* ``newton_max_iter``: maximum number of Newton iterations
* ``post_newton_fp_iter``: number of fixed-point iterations performed after each Newton
  iteraton
* ``tracer_module_names``: which tracer modules the solver is applied to

~~~~~~
Step 2
~~~~~~

Run the following command from ``$TOP`` to set up usage of the solver
::

  ./scripts/setup_solver.sh --model_name test_problem --cfg_fname <cfg_fname>

where <cfg_fname> is the path of the customized cfg file.
Running ``./scripts/setup_solver.sh -h`` shows what command line options are
available.
The ``setup_solver.sh`` script does the following:

#. Create the work directory.
   The path of the work directory, which defaults to a subdirectory of the users home
   directory, is specified by ``workdir`` in the cfg file.
   The work directory contents for test_problem are small.
#. Invoke ``gen_invoker_script``, to generate the solver's invocation script.
   The location of the solver's invocation script, which defaults to a file in the work
   directory, is specified by ``invoker_script_fname`` in the cfg file.
#. Create a vertical grid file.
   The location of the grid file, which defaults to a file in the work directory, is
   specified by ``grid_weight_fname`` in the cfg file.
   The default vertical grid has 30 layers spanning 900 m.
   The ratio of the bottom-layer thickness to the top-layer thickness is 5.0, yielding a
   top-layer thickness of 10 m.
   The defaults can be overwritten with arguments to the ``setup_solver.sh`` script.
#. Create an initial model state on the generated vertical grid.
   The initial tracer profiles are linearly interpolated to the vertical grid from the
   values specified by ``ic_vals`` and ``ic_val_depths`` in the tracer module definition
   file specified by ``tracer_module_defs_fname`` in the cfg file.
   The initial state is hard-wired to be written to a subdirectory of the work directory
   named ``gen_ic``.
#. Run the model forward from the generated initial state for a number of years.
   The number of years run, which defaults to 2, can be modified by with the ``--fp_cnt``
   argument to the ``setup_solver.sh`` script.
   The result of these forward model runs is written to the ``gen_ic`` directory.


~~~~~~
Step 3
~~~~~~

Run the invocation script generated in the previous step to start the NK solver.
Users whose default shell is not bash may need to prefix the invocation command with
``bash -i``, to ensure that conda can be invoked in invocation script.

The solver will run until a convergence criteria is met, or the maximum number of Newton
iterations is exceeded.
Both of these options are in the cfg file, as ``newton_rel_tol`` and ``newton_max_iter``
respectively.
The default settings yield convergence for the ``iage`` tracer module, but not the
``phosphorus`` tracer module.

By default, the solver solves the test_problem model with the driver persistent in memory.
The test_problem model is small enough that this is feasible.
If the ``reinvoke`` setting in the cfg file is set to True,
then the solver reinvokes itself after each forward model run and exits.
This exercises the the out-of-core functionality that is necessary for when the NK solver
is applied to large models.

The solver's progress can be monitored through examination of the solver's
:ref:`diagnostic output <solver_diagnostic_output>`.

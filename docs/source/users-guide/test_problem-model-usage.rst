.. _test_problem-model-usage:

========================
test_problem Model Usage
========================

User-configurable options for the solver and the test_problem model are in the cfg files.
The default paths of the cfg files are ``$TOP/input/test_problem/newton_krylov.cfg`` and ``$TOP/input/test_problem/model_params.cfg``, where ``$TOP`` is the toplevel directory of the repo.
Perform the following steps to spin up tracers in the test_problem model.

------
Step 1
------

Customize variable settings in the cfg files.
We recommend that the user makes these modifications on a copy of the file from the repo, to be able to preserve the settings for a particular application of the solver, and to avoid conflicts if the repo copy is updated.
The following variables are the most commonly set by the user:

* ``workdir``: directory where solver generated files are stored
* ``newton_rel_tol``: relative tolerance for Newton convergence; The solver is considered converged if :math:`|F(X)| < \text{newton_rel_tol} \cdot |X|` for each tracer module and region.
* ``newton_max_iter``: maximum number of Newton iterations
* ``post_newton_fp_iter``: number of fixed-point iterations performed after each Newton iteraton
* ``tracer_module_names``: a comma separated string of tracer modules names that the solver is applied to

------
Step 2
------

Run the following command from ``$TOP`` to set up usage of the solver
::

  ./scripts/setup_solver.sh --model_name test_problem --cfg_fnames <cfg_fname1>[,<cfg_fname2>...]

where <cfg_fnameN> are the paths of the customized cfg files.
Running ``./scripts/setup_solver.sh -h`` shows what command line options are available.
The ``setup_solver.sh`` script does the following:

#. Create the work directory.
   The path of the work directory, which defaults to a subdirectory of the users home directory, is specified by ``workdir`` in the cfg files.
   The work directory contents for test_problem are small.
#. Invoke ``gen_invoker_script``, to generate the solver's invocation script.
   The location of the solver's invocation script, which defaults to a file in the work directory, is specified by ``invoker_script_fname`` in the cfg files.
#. Create grid_vars file.
   The location of the grid file, which defaults to a file in the work directory, is specified by ``grid_vars_fname`` in the cfg files.
   The default vertical grid has 30 layers spanning 900 m.
   The ratio of the bottom-layer thickness to the top-layer thickness is 5.0, yielding a top-layer thickness of 10 m.
   The defaults can be overwritten with arguments to the ``setup_solver.sh`` script.
#. Create an initial model state on the generated vertical grid.
   The initial tracer profiles are linearly interpolated to the vertical grid from the values specified by ``init_iterate_vals`` and ``init_iterate_val_depths`` in the tracer module definition file specified by ``tracer_module_defs_fname`` in the cfg files.
   The initial state is hard-wired to be written to a subdirectory of the work directory named ``gen_init_iterate``.
#. Run the model forward from the generated initial state for a number of years.
   The number of years run, which defaults to 2, can be modified by with the ``--fp_cnt`` argument to the ``setup_solver.sh`` script.
   The result of these forward model runs is written to ``init_iterate_fname``, which is defined in the cfg files.

------
Step 3
------

Run the invocation script generated in the previous step to start the NK solver.
Users whose default shell is not bash may need to prefix the invocation command with ``bash -i``, to ensure that conda can be invoked in invocation script.

The solver will run until a convergence criteria is met, or the maximum number of Newton iterations is exceeded.
Both of these options are in the cfg files, as ``newton_rel_tol`` and ``newton_max_iter`` respectively.
The default settings yield convergence for the ``iage`` tracer module, but not the ``phosphorus`` tracer module.

By default, the solver solves the test_problem model with the driver persistent in memory.
The test_problem model is small enough that this is feasible.
If the ``reinvoke`` setting in the cfg files is set to True, then the solver reinvokes itself after each forward model run and exits.
This exercises the the out-of-core functionality that is necessary for when the NK solver is applied to large models.

The solver's progress can be monitored through examination of the solver's :ref:`diagnostic output <solver_diagnostic_output>`.

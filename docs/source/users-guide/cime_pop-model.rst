======================
cime_pop Model Details
======================

-----------
Description
-----------

The cime_pop model is the OGCM `POP <https://www.cesm.ucar.edu/models/cesm2/ocean/>`_
being run within `CESM <http://www.cesm.ucar.edu/>`_ using the coupling infrastructure of
`CIME <https://esmci.github.io/cime/versions/master/html/index.html>`_.
The rationale for this name is provided in the :ref:`FAQ <cime_pop_name_FAQ>`.
POP evolves tracers forward in time for a user specified run duration.
The function :math:`F(X)` that the Newton-Krylov (NK) solver is finding a root of is the
change in tracer that occurs when the model is run forward from state :math:`X`.
That is, the solver solves for an initial tracer state such that end state from running
the model forward in time is the same as the initial state.

-----
Usage
-----

Most options for the solver and the cime_pop model are in the cfg file.
The default location of the cfg file is ``$TOP/models/cime_pop/newton_krylov.cfg``,
where ``$TOP`` is the toplevel directory of the repo.

~~~~~~
Step 1
~~~~~~

Generate IRF tracer output for the preconditioner in the Krylov solver.

``...``

This step only needs to generate the IRF output from the IRF case.
Post-processing of the IRF output to a single file is performed in a subsequent step.

~~~~~~
Step 2
~~~~~~

Setup and build the case that the solver will use for forward model runs.

``...``

~~~~~~
Step 3
~~~~~~

Customize variable settings in the cfg file.
We recommend that the user makes these modifications on a copy of the file from the repo,
to be able to preserve the settings for a particular application of the solver, and to
avoid conflicts if the repo copy is updated.
The following variables are the most likely to need to be set by the user:

* ``workdir``: directory where solver generated files are stored
* ``newton_rel_tol``: relative tolerance for Newton convergence; The solver is considered
  converged if :math:`|F(X)| < \text{newton_rel_tol} \cdot |X|` for each tracer module
  and region.
* ``newton_max_iter``: maximum number of Newton iterations
* ``post_newton_fp_iter``: number of fixed-point iterations performed after each Newton
  iteraton
* ``tracer_module_names``: which tracer modules the solver is applied to

``...``

~~~~~~
Step 4
~~~~~~

Run the following command from ``$TOP`` to set up usage of the solver
::

  ./models/cime_pop/setup_solver.sh --cfg_fname <cfg_fname>

where <cfg_fname> is the path of the customized cfg file.
Running ``./models/cime_pop/setup_solver.sh -h`` shows what command line options are
available.
The ``setup_solver.sh`` script does the following:

#. Create the work directory.
   The path of the work directory, which defaults to
   ``/glade/scratch/$USER/newton_krylov``, is specified by ``workdir`` in the cfg file.
   This is appropriate on NCAR's cheyenne supercomputer.
   The work directory contents for cime_pop are moderate.
#. Create a time mean irf file.
   The location of the irf file, which defaults to a file in the work directory, is
   specified by ``irf_fname`` in the cfg file.
   The contents of this file are used in the preconditioner in the Krylov solver.
   Options for specifying the inputs to the mean irf file are in the cfg file.
#. Create grid weights and region files.
   The location of these files, which defaults to files in the work directory, are
   specified by ``grid_weight_fname`` and ``region_mask_fname`` in the cfg file.
   These files are generated from the irf file.
   The solver configuration function is run, to ensure that the generated files are
#. Invoke ``gen_invoker_script``, to generate the solver invoker script.
   The location of the invoker script, which defaults to a file in the work directory, is
   specified by ``invoker_script_fname`` in the cfg file.


Running the invocation script generated in the last step will start the NK solver.
The solver will run until a convergence criteria is met, or the maximum number of Newton
iterations is exceeded.
Both of these options are in the cfg file.

The cime_pop model is hard-wired to reinvoke the solver after each forward model run is
submitted to a batch job submission system.
The solver exits after submitting the job, reducing the amount of time that the solver
resides in memory.
The cime_pop model uses CIME's POSTRUN_SCRIPT feature to reinvoke the solver after the
forward model run is completed.

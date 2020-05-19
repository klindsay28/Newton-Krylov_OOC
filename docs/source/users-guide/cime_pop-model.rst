.. _users-guide-cime_pop:

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

One interpretation of how the solver works is that it performs multiple forward model runs
with perturbed tracer initial conditions, and creates an optimal initial condition
perturbation such that the difference between the tracer end state and tracer initial
state is minimized.
In order to perform forward model runs with perturbed tracer initial conditions, the
solver requires a mechanism to specify the model's tracer initial condition.
The cime_pop utilizes the xml variable ``POP_PASSIVE_TRACER_RESTART_OVERRIDE`` to do this.
The following tracer modules in CESM2 support this feature: ``iage``, ``abio_dic_dic14``,
``ecosys``.

-----
Usage
-----

Most options for the solver and the cime_pop model are in the cfg file.
The default location of the cfg file is ``$TOP/models/cime_pop/newton_krylov.cfg``,
where ``$TOP`` is the toplevel directory of the repo.
Perform the following steps to spin up tracers in the cime_pop model.

~~~~~~
Step 1
~~~~~~

Setup and build the cases that 1a) generate IRF tracer output for the preconditioner in
the Krylov solver, and 1b) the solver will use for forward model runs.
The following steps are common to both of these cases.

Note that using ``xmlchange`` is preferable to change xml variables, as opposed to editing
xml files directly, as it enables verifying that values are valid, to the extent possible.
Additionally, ``xmlchange`` commands are echoed to the CaseStatus file, creating a record
of user modifications.
This record is useful for reconstructing how a case was set up.

After the case has been created using ``create_newcase`` or ``create_clone``, set the
following xml variables for the initialization of the model: ``RUN_TYPE``,
``RUN_REFCASE``, ``RUN_REFDATE``.
The implementation of the cime_pop model in the solver assumes that ``RUN_TYPE`` is
either ``hybrid`` or ``branch``, that is, not ``startup``.
For ``RUN_TYPE=hybrid``, you also need to set ``RUN_STARTDATE``.
In CIME, this latter variable is ignored in a ``branch`` run, and the model starts on
``RUN_REFDATE``.

Prestage restart and rpointer files for ``RUN_REFCASE`` from ``RUN_REFDATE`` to
``RUNDIR``.

We recommend configuring POP to generate double precision output, to avoid unnecessary
loss of precision in the solver.
This is done by setting the xml variable ``POP_TAVG_R8`` to ``TRUE``.
We recommend doing this before running ``case.build``, as POP needs to be built, or
rebuilt, after this change is made.

The solver generates the time average of model output.
The computational cost of this step can be reduced by having the model generate annual
mean output instead of the standard monthly mean output.
This is enabled by adding the following line to ``user_nl_pop``:
::

   tavg_freq_opt(1) = 'nyear'

 Build the case by running the command ``./case.build``.
 On the NCAR/CISL machine cheyenne, CISL requests that model builds not be done on login
 nodes, to reduce computational load.
 The build can be done on batch nodes of cheyenne by running the command ``qcmd --
 ./case.build``.

Specifics for the IRF generating case that generates IRF output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the run duration by setting the following xml variables: ``STOP_OPTION``, ``STOP_N``,
``RESUBMIT``.

Run the case, to generate the IRF output, by running the command ``./case.submit``.
This step only needs to generate the IRF output from the IRF case.
Post-processing of the IRF output to a single file is performed in a subsequent step.

Specifics for the case used by solver for forward model runs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copy the rpointer files from ``RUN_REFCASE`` to a directory where they not not be
overwritten by doing a forward model run.
The solver will copy these to the run directory before each forward model run.

Ensure that the model produces for each tracer module the output fields needed to
construct the preconditioner in the Krylov solver.

~~~~~~
Step 2
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
Step 3
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
#. Invoke ``gen_invoker_script``, to generate the solver's invocation script.
   The location of the solver's invocation script, which defaults to a file in the work
   directory, is specified by ``invoker_script_fname`` in the cfg file.


~~~~~~
Step 4
~~~~~~

Run the invocation script generated in the previous step to start the NK solver.
Users whose default shell is not bash may need to prefix the invocation command with
``bash -i``, to ensure that conda can be invoked in invocation script.

The solver will run until a convergence criteria is met, or the maximum number of Newton
iterations is exceeded.
Both of these options are in the cfg file.

The cime_pop model is hard-wired to reinvoke the solver after each forward model run is
submitted to a batch job submission system.
The solver exits after submitting the job, reducing the amount of time that the solver
resides in memory.
The cime_pop model uses CIME's POSTRUN_SCRIPT feature to reinvoke the solver after the
forward model run is completed.

The solver's progress can be monitored through examination of the solver's
:ref:`diagnostic output <solver_diagnostic_output>`.

.. _add-tracer-module-cime_pop:

================================================
Adding a New Tracer Module to the cime_pop Model
================================================

The steps to add a POP tracer module to the cime_pop model of the solver are:

#. Ensure that the tracer module's implementation in POP meets the needs of the solver:

   a. Tracer Initialization: As described in the :ref:`user's guide <users-guide-cime_pop>`, the cime_pop model utilizes the xml variable ``POP_PASSIVE_TRACER_RESTART_OVERRIDE`` to specify tracer initial conditions.
      So tracer modules being supported by cime_pop need to set their namelist variables appropriately when this xml variable is not ``"none"``.
      The usage of this xml variable in POP's build-namelist script for the setting of namelist variables in the namelist ``iage_nml`` for POP's iage tracer module is a template that can be followed for other POP tracer modules.
   b. Diagnostic Output: Ensure that POP's tavg_contents file contains the fields required for the Jacobian preconditioner and additions to the solver's stats file.
      The fields required for the Jacobian preconditioner are listed in the ``hist_to_precond_varnames`` lists for each matrix definition in the ``precond_matrix_defs`` dictionary in the tracer module definition file.
      The fields required for additions to the solver's stats file are embedded in the code that adds the fields to the stats file.

#. Add metadata for the tracer module to the tracer module definition file.
   Additions to this file need to adhere to the file's format, which is described in the :ref:`user's guide <tracer-module-defn-file>`.
   These additions can be added to the copy of the tracer module definition file present in the repo at ``$TOP/input/cime_pop/newton_krylov.cfg``, or they can be added to a copy of this file, and the cfg file can point to this modified copy via the cfg file variable ``tracer_module_defs_fname``.

#. To use the tracer module that has been added to the solver, add the tracer module name to the variable ``tracer_module_names`` in the cfg file.
   Note that this is the name of the tracer module in the tracer module definition file.
   This is not necessarily the same as the name of the tracer module in POP.

----------------------------------------------------------------
Adding Tracer Module Specific Variables to the Solver Stats File
----------------------------------------------------------------

~~~~~~~~~~~~~~~~~~~~~
Tracer-like Variables
~~~~~~~~~~~~~~~~~~~~~

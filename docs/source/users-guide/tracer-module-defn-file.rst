.. _tracer-module-defn-file:

=========================================
Tracer Module Definition File Description
=========================================

The tracer module definition file is a `YAML <https://yaml.org/>`_ formatted file that provides metadata to the solver that defines the model's tracer modules and Jacobian preconditioner matrices.
The solver uses the `PyYAML <https://pyyaml.org/>`_ package to read the file content.
The file's content, represented as YAML mapping and sequence data types, is translated by the PyYAML package into python dictionary and list data types respectively.
In the following, we refer to the file's content using the terminology of the python data types.

At the top-level, the tracer module definition file is a dictionary with two key-value pairs.
One top-level key is ``tracer_module_defs`` and the corresponding value is a dictionary with tracer module definitions.
The key in each key-value pair in this dictionary is a tracer module name and the corresponding value is a dictionary defining the tracer module.
The other top-level key is ``precond_matrix_defs`` and the corresponding value is a dictionary with preconditioner matrix definitions.
The key in each key-value pair in this dictionary is a preconditioner matrix name and the corresponding value is a dictionary defining the preconditioner matrix.

------------------------------------
Tracer Module Definition Description
------------------------------------

Each tracer module is defined via a dictionary.
Two keys are recognized: ``tracers`` and ``py_mod_name``.
The value corresponding to the ``tracers`` key is a dictionary defining each tracer.
Recognized keys in the tracer definition dictionary are

* ``attrs`` The corresponding value is a dictionary of tracer attributes.
  In the test_problem model, these attributes appear in the model's history file.
  If all of the tracers in a tracer module have the same ``units`` attribute, then the corresponding value is used in the stats file for tracer module summary statistics.
* ``precond_matrix`` The corresponding value is the name of the Jacobian preconditioner matrix used for this tracer.
* ``shadows`` The corresponding value is the name of the tracer that this tracer shadows.

The test_problem model also recognizes the keys ``init_iterate_val_depths`` and ``init_iterate_vals``.
The corresponding values are lists of floats specifying depths and values respectively that are used when the ``setup_solver.sh`` script is run to generate tracer initial conditions.

The value corresponding to the ``py_mod_name`` key is the name of the module containing the python source code for the tracer module.
If this key-value pair is not specified, the solver searches for the tracer module's python source code in ``nk_ooc.model_name.tracer_module_name``.
This feature is particularly useful for parameterized tracer modules, see below, whose implementation requires tracer module specific code, but the code is independent of ``{suff}``.

.. _parameterized-tracer-module:

----------------------------
Parameterized Tracer Modules
----------------------------

Tracer module definitions can be parameterized with string.
This is represented in the tracer module name with the substring ``{suff}``.
When parameterized tracer modules are provided to ``tracer_module_names`` in the cfg files, a suffix of colon separated values for ``suff`` is also provided.
For each provided suffix, a tracer module definition is generated with ``{suff}`` replaced by the provided suffix.
If the tracers in this tracer module have a ``precond_matrix`` whose definition contains the substring ``{suff}``, this matrix definition is similarly replicated and ``{suff}`` in the matrix definition is replaced by the suffix values specified in ``tracer_module_names`` in the cfg files.

--------------------------------------------
Preconditioner Matrix Definition Description
--------------------------------------------

Before describing how preconditioner matrices are defined, we point out that there is a special matrix named ``base``, whose definition serves as a base definition for all other matrices.
Options specified in the ``base`` matrix definition are automatically propagated to all other matrices defined for the model.
An exception is for matrix definitions options specified for the ``base`` matrix and other matrices.
In this instance, options in individual matrix definitions take precedence over options in the ``base`` matrix.
This ``base`` matrix definition feature is useful for matrix definition options that apply to all matrices defined for the model, independent of the tracer to which the matrix is being applied to.
Example applications of this feature, in the context of applying the solver to spin-up tracers in an ocean model, are options describing the ocean model's circulation and mixing paramterizations, options that are applicable to all ocean model tracers.

Each preconditioner matrix, including ``base``, is defined via a dictionary.
The key ``hist_to_precond_varnames`` is utilized by all models that the solver is applied to.
The value corresponding to this key is a list of variables from model's history file that are needed to apply the Jacobian preconditioner.
The variable names in this list can have a ``:time_op`` suffix specifying an operation to be applied to reduce the variable along its time dimension.
Supported ``time_op`` values are ``mean`` and ``log_mean``.
If no ``time_op`` is specified, the history file's variable is used as during application of the preconditioner, with the exception that singleton time dimensions are dropped.
Note that the cime_pop model implementation generates a model history file by taking the time-mean of POP's tavg output, so there is an implicit ``time_op=mean`` for the cime_pop model.

The cime_pop model also utilizes the keys ``precond_matrices_opts`` and ``precond_matrices_solve_opts`` in the preconditioner matrix definition.
The value corresponding to the ``precond_matrices_opts`` key is a list of options, with arguments, provided to an `external program <https://github.com/klindsay28/NK_ocn_tracer_jacobian_precond>`_ that generates the preconditioner matrix and stores it in a netCDF file.
We note some particular options, and their sub-options, that are supported by this program:

* ``pv pv_field_name`` This option is used for POP tracers that have a surface flux with a piston velocity formulation.
  The variable ``pv_field_name`` is the name of the tracer's piston velocity in the POP tavg file.

* ``sf d_SF_d_TRACER_field_name`` This option is used for POP tracers that have a surface flux whose formulation isn't easily described with a piston velocity formulation.
  The variable ``d_SF_d_TRACER_field_name`` is the name of the variable in the POP tavg file that is the derivative of the tracer's surface flux with respect to the tracer's values, in units of cm/s.
  It is assumed that the sign convention of the surface flux is positive down.

* ``sink_type sink_opt ...`` This option is used for POP tracers that have an interior source-sink term.
  A variety of ``sink_opt`` sub-options are available for this option:

  * ``const sink_rate`` This sub-option is used for POP tracers that have a sink term with a first-order decay formulation that has a spatially constant decay rate.
    The value ``sink_rate`` is the decay rate, in units of 1/year.
    An example of a tracer with this formulation is radio-carbon, :sup:`14`\ C.

  * ``const_shallow sink_rate sink_depth`` This sub-option is for POP tracers that have a sink term with a first-order decay formulation that has a decay rate that is constant for depths shallower than ``sink_depth`` and zero for depths deeper than that.
    The value ``sink_rate`` is the decay rate, where it is non-zero, in units of 1/year.
    The units of ``sink_depth`` are cm.
    This option, with ``sink_rate = 1/hr`` and ``sink_depth = 10m`` is a good approximation for ideal age-like tracers, that are reset to a prescribed value in POP's top layer.

  * ``file sink_field_name`` This sub-option is for POP tracers that have a sink term with a first-order decay formulation that has a spatially varying decay rate.
    ``sink_field_name`` is the name of the field in the ``circ_fname`` file that provides the spatially varying decay rate.
    An example of a tracer with this formulation is dissolved organic matter that has a light dependent decay rate.

The value corresponding to the ``precond_matrices_solve_opts`` key utilized by the cime_pop model is a dictionary of dictionary of quantities used to configure batch job submission for applying the Jacobian preconditioner.
The keys of this dictionary are the POP gridname, as declared by the CIME xml variable ``OCN_GRID``.
The value corresponding to the ``OCN_GRID`` key is a dictionary with keys ``task_cnt`` and ``gigabyte_per_task``.
The values corresponding to these keys are respectively how many tasks are requested for the batch job submission, and how much memory is requested per task.
The values in the ``base`` matrix definition are appropriate for preconditioner matrices for typical single POP tracers that do not have non-local tracer coupling (e.g., like occurs with implicit sinking particles).

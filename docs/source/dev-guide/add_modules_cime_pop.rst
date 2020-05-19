.. _add_modules_cime_pop:

===========================================
Adding Tracer Modules to the cime_pop Model
===========================================

This documentation is under development.

As described in the :ref:`user's guide <users-guide-cime_pop>`, the cime_pop model
utilizes the xml variable ``POP_PASSIVE_TRACER_RESTART_OVERRIDE`` to specify
tracer initial conditions.
So tracer modules being supported by cime_pop need to set their namelist variables
appropriately when this xml variable is not "none".
The usage of this xml variable in the setting of namelist variables in ``iage_nml`` is a
template that can be followed.

------------------------------------------------
Tracer Module Definition File Format Description
------------------------------------------------

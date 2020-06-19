.. _cime_pop-model-description:

==============================
cime_pop Model Description
==============================

The cime_pop model is the OGCM `POP <https://www.cesm.ucar.edu/models/cesm2/ocean/>`_ being run within `CESM <http://www.cesm.ucar.edu/>`_ using the coupling infrastructure of `CIME <https://esmci.github.io/cime/versions/master/html/index.html>`_.
The rationale for this name is provided in the :ref:`FAQ <cime_pop_name_FAQ>`.
POP evolves tracers forward in time for a user specified run duration.
The function :math:`F(X)` that the Newton-Krylov (NK) solver is finding a root of is the change in tracer that occurs when the model is run forward from state :math:`X`.
That is, the solver solves for an initial tracer state such that end state from running the model forward in time is the same as the initial state.

One interpretation of how the solver works is that it performs multiple forward model runs with perturbed tracer initial conditions, and creates an optimal initial condition perturbation such that the difference between the tracer end state and tracer initial state is minimized.
In order to perform forward model runs with perturbed tracer initial conditions, the solver requires a mechanism to specify the model's tracer initial condition.
The cime_pop utilizes the xml variable ``POP_PASSIVE_TRACER_RESTART_OVERRIDE`` to do this.
The following POP tracer modules in CESM2 support this feature: ``iage``, ``abio_dic_dic14``, ``ecosys``.

The implementation of the cime_pop model in the solver supports the POP tracer modules ``iage`` and ``abio_dic_dic14``.
Information on adding support for other tracer modules is included in the :ref:`developer's guide <add-tracer-module-cime_pop>`.

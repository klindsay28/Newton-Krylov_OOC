====
FAQs
====

#. .. _bash_script_FAQ:

   Why is a bash script used to invoke the solver?
   A key feature of the solver is being able to restart the solver after the completion of a forward model run that might occur in a batch submission system.
   The computing environment that the forward model occurs in might not have all of the python packages required by the solver available.
   We use conda to create an environment that has the required packages.
   Putting the conda activation command and the solver invocation command inside a bash shell script is a way of ensuring that the packages are available when the solver is invoked.

#. .. _cime_pop_name_FAQ:

   Why is the cime_pop model named that, instead of cesm_pop?
   We anticipate introducing support to the solver for other climate model components, such as `MOM6 <https://github.com/NOAA-GFDL/MOM6>`_ being run within `CESM <https://github.com/NCAR/MOM6>`_.
   We also desire to include support for `MPAS-Ocean <https://github.com/MPAS-Dev/MPAS-Model>`_ being run within `E3SM <https://github.com/E3SM-Project/E3SM>`_.
   While POP in the solver is run through CESM, the interface between the solver and CESM only relies on CIME-based functionality.
   This functionality is the same functionality that would be used to support CESM/MOM6 and E3SM/MPAS-Ocean.
   So it seems natural to highlight CIME in the model name instead of CESM.
   An alternative name that is more descriptive is cime_cesm_pop.
   However, this is redundant because POP is run in CIME exclusively through CESM.

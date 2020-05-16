====
FAQs
====

.. _bash_script_FAQ:

#. Why is a bash script used to invoke the solver?
   A key feature of the solver is being able to restart the solver after the completion of
   a forward model run that might occur in a batch submission system.
   The computing environment that the forward model occurs in might not have all of the
   python packages required by the solver available.
   We use conda to create an environment that has the required packages.
   Putting the conda activation command and the solver invocation command inside a bash
   shell script is a way of ensuring that the packages are available when the solver is
   invoked.

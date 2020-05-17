=============
Getting Setup
=============

#. The solver software uses conda to create and use a software environment that includes
   the python packages used by the solver.
   If conda is not already installed for your own usage, it needs to be installed, which
   can be done using miniconda, which is `available online
   <https://docs.conda.io/en/latest/miniconda.html>`_.
#. The solver software uses a bash shell script to invoke the solver.
   As described in :ref:`FAQ <bash_script_FAQ>`, this script activates the required conda
   environment before the solver is invoked.
   So it is necessary to ensure that conda is available for yourself in bash.
   This can be verified by seeing if the command ``bash -l -c "which conda"`` runs
   successfully.
   If the output indicates that conda is not found, then you can run the command ``conda
   init bash`` to resolve this.
   This step is typically only necessary for users whose login shell is not bash.
#. Download the software from github, and create the required conda environment,
   Newton-Krylov_OOC, for using the software:
   ::

      git clone https://github.com/klindsay28/Newton-Krylov_OOC.git
      cd Newton-Krylov_OOC
      conda env create --file conda-env.yaml
#. If you have updated the file ``conda-env.yaml``, either yourself, or through updating
   Newton-Krylov_OOC, then update the conda environment:
   ::

      conda env update --file conda-env.yaml

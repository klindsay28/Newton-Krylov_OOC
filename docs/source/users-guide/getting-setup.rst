=============
Getting Setup
=============

#. Install conda for python 3, if you do not already have it installed.
   This can be done using miniconda, which is
   `available online <https://docs.conda.io/en/latest/miniconda.html>`_.
#. Download the software from github, and setup and activate the conda
   environment, Newton-Krylov_OOC, for using the software:
   ::

      git clone https://github.com/klindsay28/Newton-Krylov_OOC.git
      cd Newton-Krylov_OOC
      conda env create --file conda-env.yaml
      conda activate Newton-Krylov_OOC
#. If you have updated the file ``conda-env.yaml``, either yourself,
   or through updating Newton-Krylov_OOC, then update the conda environment:
   ::

      conda env update --file conda-env.yaml

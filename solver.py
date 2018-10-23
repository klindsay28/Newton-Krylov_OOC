"""class for representing the state of an iterative solver"""

import json
import logging
import os

class SolverState:
    """
    class for representing the state of an iterative solver

    There are no public members.

    Private members are:
    _workdir            directory where files of values are located
    _state_fname        name of file where solver state is stored
    _set_currstep       name of current step in solver
    _saved_state        dictionary of members saved and recovered across invocations
        iteration           current iteration
        step_log            steps of solver that have been logged in the current iteration
    """

    def __init__(self, workdir, state_fname, resume):
        """initialize solver state"""
        self._workdir = workdir
        self._state_fname = os.path.join(self._workdir, state_fname)
        self._currstep = None
        if resume:
            self._read_saved_state()
        else:
            self._saved_state = {'iteration':0, 'step_log':[]}

    def inc_iteration(self):
        """increment iteration, reset step_log"""
        self._saved_state['iteration'] += 1
        self._saved_state['step_log'] = []
        self._write_saved_state()
        return self._saved_state['iteration']

    def get_workdir(self):
        """return value of workdir"""
        return self._workdir

    def get_iteration(self):
        """return value of iteration"""
        return self._saved_state['iteration']

    def set_currstep(self, stepval):
        """set the value of currstep"""
        self._currstep = stepval

    def currstep_logged(self):
        """has currstep been logged in the current iteration"""
        return self._currstep in self._saved_state['step_log']

    def log_currstep(self):
        """log currstep for the current iteration"""
        self._saved_state['step_log'].append(self._currstep)
        self._write_saved_state()

    def log_saved_state(self):
        """write saved state of solver to log"""
        logger = logging.getLogger(__name__)
        logger.info('iteration=%d', self._saved_state['iteration'])
        for step_name in self._saved_state['step_log']:
            logger.info('%s completed', step_name)

    def _write_saved_state(self):
        """write _saved_state to a JSON file"""
        with open(self._state_fname, mode='w') as fptr:
            json.dump(self._saved_state, fptr, indent=2)

    def _read_saved_state(self):
        """read _saved_state from a JSON file"""
        with open(self._state_fname, mode='r') as fptr:
            self._saved_state = json.load(fptr)

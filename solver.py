"""class for representing the state of an iterative solver"""

import json
import logging
import os
import numpy as np
import util

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

        # ensure workdir exists
        util.mkdir_exist_okay(workdir)

        self._workdir = workdir
        self._state_fname = os.path.join(self._workdir, state_fname)
        self._currstep = 'init'
        if resume:
            self._read_saved_state()
        else:
            self._saved_state = {'iteration':1, 'step_log':[]}

    def get_workdir(self):
        """return value of workdir"""
        return self._workdir

    def inc_iteration(self):
        """increment iteration, reset step_log"""
        self._currstep = 'inc_iteration'
        self._saved_state['iteration'] += 1
        self._saved_state['step_log'] = []
        self._write_saved_state()
        return self._saved_state['iteration']

    def get_iteration(self):
        """return value of iteration"""
        return self._saved_state['iteration']

    def set_currstep(self, stepval):
        """set the value of currstep"""
        self._saved_state['step_log'].append(self._currstep)
        self._write_saved_state()
        self._currstep = stepval

    def get_currstep(self):
        """get the value of currstep"""
        return self._currstep

    def currstep_logged(self):
        """has currstep been logged in the current iteration"""
        return self._currstep in self._saved_state['step_log']

    def set_value_saved_state(self, key, value):
        """add a key value pair to the saved_state dictionary"""
        self._saved_state[key] = value
        self._write_saved_state()

    def get_value_saved_state(self, key):
        """get a value from the saved_state dictionary"""
        return self._saved_state[key]

    def log_saved_state(self):
        """write saved state of solver to log"""
        logger = logging.getLogger(__name__)
        logger.info('iteration=%d', self._saved_state['iteration'])
        for step_name in self._saved_state['step_log']:
            logger.info('%s completed', step_name)

    def _write_saved_state(self):
        """write _saved_state to a JSON file"""
        with open(self._state_fname, mode='w') as fptr:
            json.dump(self._saved_state, fptr, indent=2, cls=NumpyEncoder)

    def _read_saved_state(self):
        """read _saved_state from a JSON file"""
        with open(self._state_fname, mode='r') as fptr:
            self._saved_state = json.load(fptr, object_hook=json_ndarray_decode)

class NumpyEncoder(json.JSONEncoder):
    """
    extend JSONEncoder to handle numpy ndarray's
    https://stackoverflow.com/questions/26646362/nump-array-is-not-json-serializable
    """
    def default(self, o): # pylint: disable=E0202
        """method called by json.dump, when cls=NumpyEncoder"""
        if isinstance(o, np.ndarray):
            return {'__ndarray__':o.tolist()}
        return json.JSONEncoder.default(self, o)

def json_ndarray_decode(dct):
    """decode __ndarray__ tagged entries"""
    if '__ndarray__' in dct:
        return np.asarray(dct['__ndarray__'])
    return dct

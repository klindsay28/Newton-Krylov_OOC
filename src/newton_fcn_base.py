"""Base class of methods related to problem being solved with Newton's method"""

import logging
import os


class NewtonFcnBase:
    """Base class of methods related to problem being solved with Newton's method"""

    def comp_jacobian_fcn_state_prod(
        self, iterate, fcn, direction, res_fname, solver_state
    ):
        """
        compute the product of the Jacobian of fcn at iterate with the model state
        direction

        assumes direction is a unit vector
        """
        logger = logging.getLogger(__name__)
        logger.debug('res_fname="%s"', res_fname)

        fcn_complete_step = "comp_jacobian_fcn_state_prod complete for %s" % res_fname

        if solver_state.step_logged(fcn_complete_step):
            logger.debug('"%s" logged, returning result', fcn_complete_step)
            return type(iterate)(iterate.tracer_module_state_class, res_fname)
        logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        sigma = 1.0e-4 * iterate.norm()

        # perturbed ModelStateBase
        perturb_ms = iterate + sigma * direction
        perturb_fcn_fname = os.path.join(
            solver_state.get_workdir(), "perturb_fcn_" + os.path.basename(res_fname)
        )
        perturb_fcn = self.comp_fcn(  # pylint: disable=no-member
            perturb_ms, perturb_fcn_fname, solver_state
        )

        # compute finite difference
        caller = __name__ + ".NewtonFcnBase.comp_jacobian_fcn_state_prod"
        res = ((perturb_fcn - fcn) / sigma).dump(res_fname, caller)

        solver_state.log_step(fcn_complete_step)

        return res

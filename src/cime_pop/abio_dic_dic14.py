"""abio_dic_dic14 subclass of cime_pop's TracerModuleState"""

from .tracer_module_state import TracerModuleState


class abio_dic_dic14(TracerModuleState):  # pylint: disable=invalid-name
    """abio_dic_dic14 tracer module specifics for TracerModuleState"""

    def stats_vars_tracer_like(self):
        """
        return list of tracer-like vars in hist file to be processed for the stats file
        """
        res = super().stats_vars_tracer_like()
        res.append("ABIO_D14Cocn")
        return res

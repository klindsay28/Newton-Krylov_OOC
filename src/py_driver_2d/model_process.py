"""generic model process class"""


class ModelProcess:
    """generic model process class"""

    def __init__(self, depth, ypos):
        pass

    def comp_tend(self, time, tracer_vals):
        """tracer tendency from process"""
        raise NotImplementedError("Method must be implemented in derived class")

    @staticmethod
    def get_hist_vars_metadata():
        """
        return dict of process-specific history variable metadata
        return empty dict, for subclasses that do not provide this method
        """
        return {}

    def hist_write(self, sol, fptr_hist):
        """
        write processs-specific history variables
        empty stub provided to support subclasses that do not provide this method
        """

    def comp_jacobian(self, time):
        """compute jacobian of tracer tendencies from process"""
        raise NotImplementedError("Method must be implemented in derived class")

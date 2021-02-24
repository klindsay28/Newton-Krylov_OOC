"""test functions in model_config.py"""

from src.model_config import propagate_base_matrix_defs_to_all

from .share import config_test_problem


def test_model_config():
    """
    configure the test_problem model and confirm that the returned model_config_obj has
    expected attributes
    """

    model_config_obj = config_test_problem()

    # confirm that model_config_obj has particular attributes
    assert hasattr(model_config_obj, "modelinfo")
    assert hasattr(model_config_obj, "tracer_module_defs")
    assert hasattr(model_config_obj, "precond_matrix_defs")
    assert hasattr(model_config_obj, "region_mask")
    assert hasattr(model_config_obj, "grid_weight")


def test_propagate_base_matrix_defs_to_all():
    """test propagate_base_matrix_defs_to_all"""

    model_config_obj = config_test_problem()

    precond_matrix_defs = model_config_obj.precond_matrix_defs

    base_def = precond_matrix_defs["base"]
    phosphorus = precond_matrix_defs["phosphorus"]

    # verify that base_def's hist_to_precond_varnames were propagated to phosphorus
    for varname in base_def["hist_to_precond_varnames"]:
        assert varname in phosphorus["hist_to_precond_varnames"]

    # add a hist var to base, re-propagate, and verify that it is added to phosphorus
    base_def["hist_to_precond_varnames"].append("new_hist_var")
    propagate_base_matrix_defs_to_all(precond_matrix_defs)
    assert "new_hist_var" in phosphorus["hist_to_precond_varnames"]

    # add a matrix opt to base, re-propagate, and verify that it is added to phosphorus
    base_def["precond_matrices_opts"] = ["matrix_opt_A sub_opt"]
    propagate_base_matrix_defs_to_all(precond_matrix_defs)
    assert "precond_matrices_opts" in phosphorus
    assert "matrix_opt_A sub_opt" in phosphorus["precond_matrices_opts"]

    # add a matrix opt base and phosphorus matrices, with different sub_opts,
    # re-propagate, and verify that phosphorus sub_opt doesn't change
    # verify that matrix_opt_A wasn't added again
    base_def["precond_matrices_opts"].append("matrix_opt_B sub_opt_base")
    phosphorus["precond_matrices_opts"].append("matrix_opt_B sub_opt_phosphorus")
    propagate_base_matrix_defs_to_all(precond_matrix_defs)
    assert "precond_matrices_opts" in phosphorus
    assert "matrix_opt_B sub_opt_phosphorus" in phosphorus["precond_matrices_opts"]
    assert "matrix_opt_B sub_opt_base" not in phosphorus["precond_matrices_opts"]
    assert phosphorus["precond_matrices_opts"].count("matrix_opt_A sub_opt") == 1

"""vars and methods common to comp_fcn.py and comp_precond_jacobian_fcn_state_prod.py"""

import numpy as np

t_beg = 0.0
t_end = 365.0
t_del = t_end - t_beg

nz = 50

z_edge = np.linspace(0.0, 500.0, 1+nz)
z_mid = 0.5 * (z_edge[:-1] + z_edge[1:])

dz = np.ediff1d(z_edge)
dz_r = 1.0 / dz

dz_mid_r = 1.0 / np.ediff1d(z_mid)

def mixing_coeff(time):
    """return vertical mixing coefficient, m2 d-1"""

    bldepth_min = 50.0
    bldepth_max = 150.0
    bldepth_del = bldepth_max - bldepth_min
    bldepth = bldepth_min \
        + bldepth_del * (0.5 + 0.5 * np.cos((2 * np.pi) * ((time / 365.0) - 0.25)))
    # z_lin ranges from 0.0 to 1.0 over span of 50.0 m, is 0.5 at bldepth
    z_lin = np.maximum(0.0, np.minimum(1.0, 0.5 + (z_edge - bldepth) * (1.0 / 50.0)))
    res_log10_shallow = 0.0
    res_log10_deep = -5.0
    res_log10_del = res_log10_deep - res_log10_shallow
    res_log10 = res_log10_shallow + res_log10_del * z_lin
    return 86400 * (10.0 ** res_log10)

def mixing_coeff_avg():
    """
    return time average of mixing_coeff over the interval t_beg:t_end
    compute using midpoint rule with samps intervals
    """
    res = np.zeros(1+nz)
    samps = 100
    t_del_samp = t_del / samps
    for t_ind in range(samps):
        res += mixing_coeff(t_beg + (t_ind + 0.5) * t_del_samp)
    return (1.0 / samps) * res

def mixing_coeff_log_avg():
    """
    return time average of mixing_coeff over the interval t_beg:t_end
    compute using midpoint rule with samps intervals
    """
    res = np.zeros(1+nz)
    samps = 100
    t_del_samp = t_del / samps
    for t_ind in range(samps):
        res += np.log(mixing_coeff(t_beg + (t_ind + 0.5) * t_del_samp))
    return np.exp((1.0 / samps) * res)

def mixing_coeff_max():
    """
    return max over time of mixing_coeff over the interval t_beg:t_end
    """
    res = np.zeros(1+nz)
    samps = 100
    t_del_samp = t_del / samps
    for t_ind in range(samps):
        res = np.maximum(res, mixing_coeff(t_beg + (t_ind + 0.5) * t_del_samp))
    return res

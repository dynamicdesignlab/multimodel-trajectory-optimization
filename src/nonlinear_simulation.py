import numpy as np
import scipy.integrate
import time

TRACK_EDGE_OVERRUN_MARGIN = 1 # Maximum allowable track bound exceedance before stopping simulation


def simulate(stm, world, s_ref, x_ref, u_ref, z_ref, fb_gains_dict, s_sim, mu_road_sim, x_0=None):
    """
    Simulate the closed-loop dynamics of trajectory tracking to the reference.
    
    s_ref, x_ref, u_ref, and z_ref are the reference signals
    fb_gains_dict is the dictionary of tracking feedback gains
    s_sim is the desired vector to evaluate the solution on
    mu_road_sim is the road friction coefficient on the same grid as s_sim
    x_0 is the initial condition
    """
        
    # Construct interpolants for the integrator
    x_ref_interpolant = scipy.interpolate.interp1d(s_ref, x_ref, axis=1, kind="linear", bounds_error=False, assume_sorted=True, fill_value="extrapolate")
    u_ref_interpolant = scipy.interpolate.interp1d(s_ref, u_ref, axis=1, kind="linear", bounds_error=False, assume_sorted=True, fill_value="extrapolate")
    z_ref_interpolant = scipy.interpolate.interp1d(s_ref, z_ref, axis=1, kind="linear", bounds_error=False, assume_sorted=True, fill_value="extrapolate")
    mu_road_interpolant = scipy.interpolate.interp1d(s_sim, mu_road_sim, fill_value="extrapolate")
    track_width_interpolant = world.virtual_track_width_m_interp_fcn
    world_length = world.length_m
    
    # Define an event for exceeding the bounds of the track
    def _max_lateral_error_event(s,
                                 x_sim,
                                 x_ref_interpolant,
                                 u_ref_interpolant,
                                 z_ref_interpolant,
                                 track_width_interpolant,
                                 fb_gains_dict,
                                 mu_road_interpolant,
                                 world_length):
        
        e_max_m = track_width_interpolant(s)/2 + TRACK_EDGE_OVERRUN_MARGIN
        return np.abs(x_sim[4]) - e_max_m
    
    _max_lateral_error_event.terminal = True
    _max_lateral_error_event.direction = 0
        
    start_time = time.time()
    
    if(x_0 is None): x_0 = x_ref[:, 0]
    
    # Simulate the closed-loop dynamics
    ivp_sol = scipy.integrate.solve_ivp(fun=stm.dx_ds_sim,
                                        t_span=(s_sim[0], s_sim[-1]),
                                        y0=x_0,
                                        method="RK45",
                                        events=[_max_lateral_error_event],
                                        t_eval=s_sim,
                                        args=(x_ref_interpolant,
                                              u_ref_interpolant,
                                              z_ref_interpolant,
                                              track_width_interpolant,
                                              fb_gains_dict,
                                              mu_road_interpolant,
                                              world_length),
                                        )
    
    # Print solve and maneuver times
    maneuver_time = np.inf
    if(ivp_sol["success"]==True and np.abs(ivp_sol["t"][-1]-s_sim[-1])<=0.001): maneuver_time = ivp_sol["y"][3][-1]
    print("Finished simulation in {} sec; t_final = {} sec; s_final = {} m (out of {} m)".format(round(time.time() - start_time, 3), round(maneuver_time, 3), round(ivp_sol["t"][-1], 3), round(s_sim[-1], 3)))
    
    
    # Back-calculate u_sim based on feedforward + feedback law from solution
    s_sim_sol = ivp_sol["t"]
    x_sim_sol = ivp_sol["y"]
    u_sim_sol = np.zeros((2, x_sim_sol.shape[1]))
    delta_max_rad = np.radians(stm.params_dict["max_delta_deg"])
    
    delta_fb = - (  fb_gains_dict["K_delta_e"]*(x_sim_sol[4, :] - x_ref_interpolant(s_sim_sol % world_length)[4, :])
                  + fb_gains_dict["K_delta_dpsi"]*(x_sim_sol[5, :] - x_ref_interpolant(s_sim_sol % world_length)[5, :]))
    
    fx_fb = - (fb_gains_dict["K_fx_ux"]*(x_sim_sol[0, :] - x_ref_interpolant(s_sim_sol % world_length)[0, :]))
    
    u_sim_sol = u_ref_interpolant(s_sim_sol % world_length)
    u_sim_sol[0, :] += delta_fb
    u_sim_sol[1, :] += fx_fb
    u_sim_sol[0, :] = np.clip(u_sim_sol[0, :], -delta_max_rad, delta_max_rad)
    
    return s_sim_sol, x_sim_sol, u_sim_sol
import trajectory_planning_helpers
import numpy as np


def plan_trajectory(stm, world, params_dict, plot_results=True):

    sol_dict = {} # Allocate a dictionary to contain the solutions
    sol_dict["params_dict_opt"] = params_dict
    sol_dict["params_dict_stm"] = stm.params_dict

    
    # Specify the start point of the horizon depending on the value of the CONVERGENT_LAP flag
    if(params_dict["CONVERGENT_LAP"]):
        x_0 = np.array([None, None, None, 0, None, None, None]) # vx, vy, r, t, e, dpsi, dfz_long
        u_0 = np.array([None, None]) # delta, fx
    else:
        x_0 = np.array([None, 0, 0, 0, None, 0, 0])
        u_0 = np.array([0, 0])
        
    
    # Define a simple warmstart for the optimization
    s_vals = np.linspace(params_dict["S_START_M"], params_dict["S_END_M"], params_dict["N"]+1)
    x_warmstart = np.zeros((7, params_dict["N"] + 1))
    u_warmstart = np.zeros((2, params_dict["N"] + 1))
    
    x_warmstart[0, :] = 5 # Constant speed
    x_warmstart[3, :] = np.cumsum(params_dict["DS_M"]/x_warmstart[0, :]) # Finite sum of time
    x_warmstart[3, :] -= x_warmstart[3, 0] # Start time of zero    
    
    # Single-model trajectory optimization
    if(len(params_dict["MU_F_OPT"])==1):
        
        print("Starting single-model trajectory optimization...")
        sm_trajectory_sol = trajectory_planning_helpers.plan_single_model_trajectory(
                                                            stm,
                                                            world,
                                                            params_dict["N"],
                                                            params_dict["DS_M"],
                                                            params_dict["MU_F_OPT"][0],
                                                            params_dict["MU_R_OPT"][0],
                                                            x_0=x_0,
                                                            u_0=u_0,
                                                            s_0=params_dict["S_START_M"],
                                                            weight_delta_dot=params_dict["WEIGHT_DELTA_DOT"],
                                                            weight_fx_dot=params_dict["WEIGHT_FX_DOT"],
                                                            edge_buffer=params_dict["TRACK_EDGE_BUFFER_M"],
                                                            x_warmstart=x_warmstart,
                                                            u_warmstart=u_warmstart,
                                                            convergent_lap=params_dict["CONVERGENT_LAP"],
                                                            vx_max_mps=params_dict["VX_MAX_MPS"])
        
        sol_dict["trajectory_0_sol"] = sm_trajectory_sol
    
    
    # Multi-model trajectory optimization
    elif(len(params_dict["MU_F_OPT"])==2):      
        
        print("Starting multi-model trajectory optimization...")
        mm_trajectory_0_sol, mm_trajectory_1_sol = trajectory_planning_helpers.plan_multi_model_trajectory(
                                                            stm,
                                                            world,
                                                            params_dict["N"],
                                                            params_dict["DS_M"],
                                                            params_dict["MU_F_OPT"],
                                                            params_dict["MU_R_OPT"],
                                                            x_0=x_0,
                                                            u_0=u_0,
                                                            s_0=params_dict["S_START_M"],
                                                            weight_delta_dot=params_dict["WEIGHT_DELTA_DOT"],
                                                            weight_fx_dot=params_dict["WEIGHT_FX_DOT"],
                                                            fb_gains_dict=params_dict["FEEDBACK_GAINS_DICT"],
                                                            edge_buffer=params_dict["TRACK_EDGE_BUFFER_M"],
                                                            x_warmstart=x_warmstart,
                                                            u_warmstart=u_warmstart,
                                                            convergent_lap=params_dict["CONVERGENT_LAP"],
                                                            vx_max_mps=params_dict["VX_MAX_MPS"])
        
        sol_dict["trajectory_0_sol"] = mm_trajectory_0_sol
        sol_dict["trajectory_1_sol"] = mm_trajectory_1_sol
        
    else:
        raise ValueError("Length of params_dict['MU_F_OPT'] must be either 1 or 2 (for single- or multi-model).")       
    
    return sol_dict
import datetime
from world import World
import matplotlib.pyplot as plt
import numpy as np
import plotting_utils
import nonlinear_simulation
import pickle
from single_track_model import SingleTrackModel
import trajectory_planning

if(__name__=="__main__"):
    
    plt.close("all")    
    
    # %% Set parameters for the world map and trajectory optimization
    params_dict = {
        
        # Vehicle model parameters
        "VEHICLE_PARAMS_FILENAME" : "vehicle_params_gti.yaml",
        
        # Map parameters
        "MAP_FILENAME"   : "Medium_Oval_Map_260m.mat",
        "MAP_NAME"       : "Medium_Oval",
        "SECTOR_NAME"    : "Full_Lap",
        "CONVERGENT_LAP" : True, # Whether to solve for a coincident horizon start and end point
        "S_START_M"      : 0, # Horizon start point s-coordinate
        "S_END_M"        : 260, # Horizon end point s-coordinate
        "N"              : 260, # Number of discretization steps        
        
        # Trajectory optimization parameters
        "WEIGHT_DELTA_DOT" : 5,
        "WEIGHT_FX_DOT"    : 5,                           
        
        "FEEDBACK_GAINS_DICT" : {"K_delta_e": 0.18, "K_delta_dpsi": 1.50, "K_fx_ux": 2.00},
    
        "VX_MAX_MPS"          : None, # Speed limit; set to None if not desired
        "TRACK_EDGE_BUFFER_M" : 0.0, # Edge buffer to road bounds for motion planning (Must be >= 0)
    }

    params_dict["DS_M"] = (params_dict["S_END_M"] - params_dict["S_START_M"])/params_dict["N"] # Calculate the derived horizon step size
    
    

    # %% Load in vehicle and map model based on user specifications
    niki_stm = SingleTrackModel(params_dict["VEHICLE_PARAMS_FILENAME"])
    
    world = World(filename=params_dict["MAP_FILENAME"],
                  name = params_dict["MAP_NAME"],
                  edge_buffer_m = params_dict["TRACK_EDGE_BUFFER_M"],
                  diagnostic_plotting=True)
    

    # %% Specify the desired trajectory optimization type
    
    # Select from the following options:
    # - SINGLE_MODEL_LOWER_BOUND_MU
    # - SINGLE_MODEL_UPPER_BOUND_MU
    # - MULTI_MODEL
        
    OPT_TYPE = "MULTI_MODEL" 
    
    
    # %% Configure friction coefficient parameters according to the selected trajectory optimization type
    if(OPT_TYPE=="SINGLE_MODEL_LOWER_BOUND_MU"):
        mu_f_opt = [niki_stm.params_dict["mu_f_lb"]]
        mu_r_opt = [niki_stm.params_dict["mu_r_lb"]]
    
    elif(OPT_TYPE=="SINGLE_MODEL_UPPER_BOUND_MU"):
        mu_f_opt = [niki_stm.params_dict["mu_f_ub"]]
        mu_r_opt = [niki_stm.params_dict["mu_f_ub"]]
    
    elif(OPT_TYPE=="MULTI_MODEL"):
        mu_f_opt = [niki_stm.params_dict["mu_f_ub"], niki_stm.params_dict["mu_f_lb"]]
        mu_r_opt = [niki_stm.params_dict["mu_r_ub"], niki_stm.params_dict["mu_r_lb"]]
        
    else:
        raise ValueError("Please specify 'OPT_TYPE' as one of either 'SINGLE_MODEL_LOWER_BOUND_MU', 'SINGLE_MODEL_UPPER_BOUND_MU', or 'MULTI_MODEL'.")
        
    params_dict["MU_F_OPT"] = mu_f_opt
    params_dict["MU_R_OPT"] = mu_r_opt
    

    
    # %% Run the trajectory optimization
    results = trajectory_planning.plan_trajectory(niki_stm, world, params_dict)
    
    
    # %% Simulate the closed-loop with different road friction profiles
        
    ds_sim = 0.10 # Desired spacing of simulation [m]
    s_sim = np.arange(params_dict["S_START_M"], params_dict["S_END_M"]+ds_sim, ds_sim)
        
    mu_vals = np.arange(0.10, 0.35+0.05, 0.05) # Friction coefficients to simulate
    sim_suite = [mu_vals[i]*np.ones_like(s_sim) for i in range(len(mu_vals))]


    # Run a nonlinear simulation for each mu profile in the sim_suite
    for i, mu_sim in enumerate(sim_suite):
        
        if("SINGLE_MODEL" in OPT_TYPE):
            x_0 = results["trajectory_0_sol"]["X"][:, 0]
        else:
            x_0 = results["trajectory_1_sol"]["X"][:, 0]
            
        s_sim_sol, x_sim_sol, u_sim_sol = nonlinear_simulation.simulate(
            niki_stm,
            world,
            results["trajectory_0_sol"]["s_m"],
            results["trajectory_0_sol"]["X"],
            results["trajectory_0_sol"]["U"],
            results["trajectory_0_sol"]["Z"],
            params_dict["FEEDBACK_GAINS_DICT"],
            s_sim,
            mu_sim,
            x_0)
        
        sim_dict = {}
        sim_dict["s_sim"] = s_sim_sol
        sim_dict["mu_road_sim"] = mu_sim
        sim_dict["x_sim"] = x_sim_sol
        sim_dict["u_sim"] = u_sim_sol
        sim_dict["east_m"], sim_dict["north_m"], sim_dict["up_m"] = world.map_match_vectorized(s_sim_sol, x_sim_sol[4, :])
        results["simulation_"+str(i)] = sim_dict
        
    plotting_utils.plot_results(results)
    
    
    # %% Save results in a pickle file
    
    results["world"] = world.data
    
    now = datetime.datetime.now()
    now_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    with open("results/" + "results_" + now_string + ".pkl", "wb") as f:
        pickle.dump(results, f)
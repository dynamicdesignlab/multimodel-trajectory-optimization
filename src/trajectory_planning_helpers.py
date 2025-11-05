PI = 3.14159265359

from casadi import Opti, cos
import numpy as np


def plan_single_model_trajectory(stm_model,
                                 world,
                                 N,
                                 ds_m,
                                 mu_f_opt,
                                 mu_r_opt,
                                 x_0,
                                 u_0,
                                 s_0,
                                 weight_delta_dot, weight_fx_dot,
                                 edge_buffer=None,
                                 x_warmstart=None,
                                 u_warmstart=None,
                                 convergent_lap=False,
                                 vx_max_mps=None):
    opti = Opti()

    # Define decision variables
    X = opti.variable(7, N+1) # state evolution
    U = opti.variable(2, N+1) # control effort evolution
    
    VX         = X[0, :] # body longitudinal velocity [m/s]
    VY         = X[1, :] # body lateral velocity [m/s]
    R          = X[2, :] # body yaw rate [rad/s]
    T          = X[3, :] # time along horizon [s]
    E          = X[4, :] # lateral distance from local track centerline [m]
    DPSI       = X[5, :] # heading error to local heading of track [rad]
    DFZ_LONG   = X[6, :] # longitudinal weight transfer [kN]
    
    DELTA      = U[0, :] # steering angle [rad]
    FX         = U[1, :] # vehicle longitudinal force input [kN]


    # Define auxiliary variables, which help with writing constraints
    AUX = opti.variable(12, N+1)

    FX_F = AUX[0, :]             # Front longitudinal tire force [kN]
    FY_F = AUX[1, :]             # Front lateral tire force [kN]
    FX_R = AUX[2, :]             # Rear longitudinal tire force [kN]
    FY_R = AUX[3, :]             # Rear lateral tire force [kN]
    FZ_F = AUX[4, :]             # Front axle normal force [kN]
    FZ_R = AUX[5, :]             # Rear axle normal force [kN]
    AX = AUX[6, :]               # Longitudinal body acceleration [m/s^2]
    AY = AUX[7, :]               # Lateral body acceleration [m/s^2]
    ALPHA_Z = AUX[8, :]          # Rotational body acceleration [rad/s^2]
    ALPHA_F = AUX[9, :]          # Front axle slip angle [rad]
    ALPHA_R = AUX[10, :]         # Rear axle slip angle [rad]
    MZ_DIFF_BRAKING = AUX[11, :] # Yaw moment due to differential braking [kN-m]


    # Pre-compute track topography for the whole horizon
    S = np.linspace(s_0, s_0 + N*ds_m, N+1)
    Z = np.zeros((6, N+1))
    Z[0, :] = np.ravel(np.asarray(world.psi_rad_LUT(S)))
    Z[1, :] = np.ravel(np.asarray(world.grade_rad_LUT(S)))
    Z[2, :] = np.ravel(np.asarray(world.bank_rad_LUT(S)))
    Z[3, :] = np.ravel(np.asarray(world.psi_s_radpm_LUT(S)))
    Z[4, :] = np.ravel(np.asarray(world.grade_s_radpm_LUT(S)))
    Z[5, :] = np.ravel(np.asarray(world.bank_s_radpm_LUT(S)))


    # Define the objective function
    W_time     = (20/(N*ds_m))**2 # Time weight
    W_deltadot = weight_delta_dot*(1/(stm_model.params_dict["max_delta_dot_degps"]*PI/180))**2 # Steering slew rate weight
    W_fxdot    = weight_fx_dot   *(1/(stm_model.params_dict["max_fx_dot_nps"]/1000))**2 # Longitudinal force command rate weight
    
    cost = 0
    cost += W_time * T[-1]**2 # Final horizon time cost
    for k in range(1, N+1):
        cost += W_deltadot * ((U[0, k] - U[0, k-1])/(T[k] - T[k-1]))**2/(N+1) # Steering slew rate cost
        cost += W_fxdot    * ((U[1, k] - U[1, k-1])/(T[k] - T[k-1]))**2/(N+1) # Longitudinal force input rate cost
        
    opti.minimize(cost)


    # Define constraints
    
    # Track bounds
    for k in range(0, N+1):
        opti.subject_to(opti.bounded(-(world.track_width_m_LUT(S[k])/2 - edge_buffer),
                                      E[k],
                                      (world.track_width_m_LUT(S[k])/2 - edge_buffer)))
            

    # Dynamics constraints (trapezoidal integration)
    for k in range(N):
        
        # Compute state derivative at left knot point
        f_left, FX_F[:, k], FY_F[:, k], FX_R[:, k], FY_R[:, k], FZ_F[:, k], FZ_R[:, k], \
        AX[:, k], AY[:, k], ALPHA_Z[:, k], ALPHA_F[:, k], ALPHA_R[:, k], MZ_DIFF_BRAKING[:, k] \
            = stm_model.dx_ds_casadi(X[:, k], U[:, k], Z[:, k], mu_f_opt, mu_r_opt, auxiliary_outputs=True) 
        
        # Compute state derivative at right knot point
        f_right, FX_F[:, k+1], FY_F[:, k+1], FX_R[:, k+1], FY_R[:, k+1], FZ_F[:, k+1], FZ_R[:, k+1], \
        AX[:, k+1], AY[:, k+1], ALPHA_Z[:, k+1], ALPHA_F[:, k+1], ALPHA_R[:, k+1], MZ_DIFF_BRAKING[:, k+1] \
            = stm_model.dx_ds_casadi(X[:, k+1], U[:, k+1], Z[:, k+1], mu_f_opt, mu_r_opt, auxiliary_outputs=True)

        # Dynamics evolution constraint
        opti.subject_to(X[:, k+1] - X[:, k] - 1/2*ds_m*(f_left + f_right) == 0)


    # Actuator constraints
    opti.subject_to(opti.bounded(-(PI/180)*stm_model.params_dict["max_delta_deg"], DELTA[:], (PI/180)*stm_model.params_dict["max_delta_deg"])) # Steering angle limit
    opti.subject_to(FX[:]*VX[:] <= stm_model.params_dict["p_eng_max_kw"]) # Engine power limit
    opti.subject_to(opti.bounded(-mu_f_opt*FZ_F[:]*cos(ALPHA_F[:]), FX_F[:], mu_f_opt*FZ_F[:]*cos(ALPHA_F[:]))) # Constrain Fx_f command to friction circle
    opti.subject_to(opti.bounded(-mu_r_opt*FZ_R[:]*cos(ALPHA_R[:]), FX_R[:], mu_r_opt*FZ_R[:]*cos(ALPHA_R[:]))) # Constrain Fx_r command to friction circle
    if(vx_max_mps is not None): opti.subject_to(VX[:] <= vx_max_mps) # Speed limit



    # Boundary conditions
    if(convergent_lap):
        opti.subject_to(VX[0]        == VX[-1])
        opti.subject_to(VY[0]        == VY[-1])
        opti.subject_to(R[0]         == R[-1])
        opti.subject_to(T[0]         == x_0[3])
        opti.subject_to(E[0]         == E[-1])
        opti.subject_to(DPSI[0]      == DPSI[-1])
        opti.subject_to(DFZ_LONG[0]  == DFZ_LONG[-1])

        opti.subject_to(DELTA[0]     == DELTA[-1])
        opti.subject_to(FX[0]        == FX[-1])

    else:
        raise NotImplementedError("Trajectory optimization currently is configured for full convergent laps.")


    # Initialize solver
    opti.set_initial(X, x_warmstart)
    opti.set_initial(DELTA[:], u_warmstart[0, :])
    opti.set_initial(FX[:], u_warmstart[1, :])


    ###########################################################################


    # Setup and call optimization

    # Solver setup
    p_opts = {"expand": True}
    s_opts = {"print_level" : 5, "max_iter" : 5000}
    opti.solver("ipopt", p_opts, s_opts)
    
    # Call optimization routine
    sol = opti.solve()

    # Process results
    X_sol = sol.value(X)
    U_sol = sol.value(U)
    Z_sol = sol.value(Z)
    
    VX_sol = sol.value(VX)
    VY_sol = sol.value(VY)
    R_sol = sol.value(R)
    T_sol = sol.value(T)
    E_sol = sol.value(E)
    DPSI_sol = sol.value(DPSI)
    DFZ_LONG_sol = sol.value(DFZ_LONG)

    DELTA_sol = sol.value(DELTA)
    FX_sol = sol.value(FX)

    FX_F_sol = sol.value(FX_F)
    FY_F_sol = sol.value(FY_F)
    FX_R_sol = sol.value(FX_R)
    FY_R_sol = sol.value(FY_R)
    FZ_F_sol = sol.value(FZ_F)
    FZ_R_sol = sol.value(FZ_R)

    AX_sol = sol.value(AX)
    AY_sol = sol.value(AY)
    ALPHA_Z_sol = sol.value(ALPHA_Z)
    ALPHA_F_sol = sol.value(ALPHA_F)
    ALPHA_R_sol = sol.value(ALPHA_R)
    MZ_DIFF_BRAKING_sol = sol.value(MZ_DIFF_BRAKING)

    S_sol = np.linspace(s_0, s_0 + N*ds_m, N+1)
    
    print(f"Solution maneuver time: {T_sol[-1]:0.02f}")
    print(f"Maximum acceleration: {1/9.81*np.max(np.sqrt(AX_sol[:]**2 + AY_sol[:]**2)):0.02f} g's")
    
    x_sol, y_sol, z_sol = world.map_match_vectorized(S_sol, E_sol)

    posE_true_innerbound = world.inner_bounds_posE_m_interp_fcn(S_sol)
    posN_true_innerbound = world.inner_bounds_posN_m_interp_fcn(S_sol)
    posU_true_innerbound = world.inner_bounds_posU_m_interp_fcn(S_sol)
    posE_true_outerbound = world.outer_bounds_posE_m_interp_fcn(S_sol)
    posN_true_outerbound = world.outer_bounds_posN_m_interp_fcn(S_sol)
    posU_true_outerbound = world.outer_bounds_posU_m_interp_fcn(S_sol)
    
    posE_virtual_innerbound = world.virtual_inner_bounds_posE_m_interp_fcn(S_sol)
    posN_virtual_innerbound = world.virtual_inner_bounds_posN_m_interp_fcn(S_sol)
    posU_virtual_innerbound = world.virtual_inner_bounds_posU_m_interp_fcn(S_sol)
    posE_virtual_outerbound = world.virtual_outer_bounds_posE_m_interp_fcn(S_sol)
    posN_virtual_outerbound = world.virtual_outer_bounds_posN_m_interp_fcn(S_sol)
    posU_virtual_outerbound = world.virtual_outer_bounds_posU_m_interp_fcn(S_sol)

    
    trajectory_0 = {
        "X":                         X_sol,
        "U":                         U_sol,
        "Z":                         Z_sol,
        "vx_mps":                    VX_sol,
        "vy_mps":                    VY_sol,
        "r_radps":                   R_sol,
        "t_s":                       T_sol,
        "e_m":                       E_sol,
        "dpsi_rad":                  DPSI_sol,
        "dfz_long_kn":               DFZ_LONG_sol,
        "x_m":                       x_sol,
        "y_m":                       y_sol,
        "z_m":                       z_sol,
        "s_m":                       S_sol,
        "delta_rad":                 DELTA_sol,
        "fx_kn":                     FX_sol,
        "fx_f_kn":                   FX_F_sol,
        "fy_f_kn":                   FY_F_sol,
        "fx_r_kn":                   FX_R_sol,
        "fy_r_kn":                   FY_R_sol,
        "fz_f_kn":                   FZ_F_sol,
        "fz_r_kn":                   FZ_R_sol,
        "mz_diff_braking_knm":       MZ_DIFF_BRAKING_sol,
        "ax_mps2":                   AX_sol,
        "ay_mps2":                   AY_sol,
        "alpha_z_radps2":            ALPHA_Z_sol,
        "alpha_f_rad":               ALPHA_F_sol,
        "alpha_r_rad":               ALPHA_R_sol,
        "posE_m_true_innerbound":    posE_true_innerbound,
        "posN_m_true_innerbound":    posN_true_innerbound,
        "posU_m_true_innerbound":    posU_true_innerbound,
        "posE_m_true_outerbound":    posE_true_outerbound,
        "posN_m_true_outerbound":    posN_true_outerbound,
        "posU_m_true_outerbound":    posU_true_outerbound,
        "posE_m_virtual_innerbound": posE_virtual_innerbound,
        "posN_m_virtual_innerbound": posN_virtual_innerbound,
        "posU_m_virtual_innerbound": posU_virtual_innerbound,
        "posE_m_virtual_outerbound": posE_virtual_outerbound,
        "posN_m_virtual_outerbound": posN_virtual_outerbound,
        "posU_m_virtual_outerbound": posU_virtual_outerbound,
        "stm_model_params":          stm_model.params_dict,
        "mu_f_opt":                  mu_f_opt,
        "mu_r_opt":                  mu_r_opt,
        "final_t_s":                 T_sol[-1]
    }


    return trajectory_0




def plan_multi_model_trajectory(stm_model,
                                 world,
                                 N,
                                 ds_m,
                                 mu_f_opt,
                                 mu_r_opt,
                                 x_0,
                                 u_0,
                                 s_0,
                                 weight_delta_dot, weight_fx_dot,
                                 fb_gains_dict,
                                 edge_buffer=None,
                                 x_warmstart=None,
                                 u_warmstart=None,
                                 convergent_lap=False,
                                 vx_max_mps=None):
    """
    Multi-model trajectory optimization.

    Trajectory 0: Nominal trajectory
    Trajectory 1: Auxiliary trajectory
    """


    opti = Opti()

    # Define decision variables
    X0 = opti.variable(7, N+1) # state evolution, model 0
    X1 = opti.variable(7, N+1) # state evolution, model 1

    U0  = opti.variable(2, N+1) # control effort evolution, model 0
    U1  = opti.variable(2, N+1) # control effort evolution, model 1    
    
    # Define auxiliary variables
    AUX0 = opti.variable(12, N+1)
    AUX1 = opti.variable(12, N+1)
    
    opti.subject_to(AUX1[0, :] <= 20)
    
    # Pre-compute track topograpy for the horizon
    S = np.linspace(s_0, s_0 + N*ds_m, N+1)
    Z = np.zeros((6, N+1))
    Z[0, :] = np.ravel(np.asarray(world.psi_rad_LUT(S)))
    Z[1, :] = np.ravel(np.asarray(world.grade_rad_LUT(S)))
    Z[2, :] = np.ravel(np.asarray(world.bank_rad_LUT(S)))
    Z[3, :] = np.ravel(np.asarray(world.psi_s_radpm_LUT(S)))
    Z[4, :] = np.ravel(np.asarray(world.grade_s_radpm_LUT(S)))
    Z[5, :] = np.ravel(np.asarray(world.bank_s_radpm_LUT(S)))


    # Define the objective function    
    W_time     = (20/(N*ds_m))**2 # Time weight
    W_deltadot = weight_delta_dot*(1/(stm_model.params_dict["max_delta_dot_degps"]*PI/180))**2
    W_fxdot    = weight_fx_dot   *(1/(stm_model.params_dict["max_fx_dot_nps"]/1000))**2

    cost = 0
    cost += W_time*(1/2)*(X0[3, -1]**2 + X1[3, -1]**2)
    for k in range(1, N+1):
        cost += W_deltadot*(1/2)*((U1[0, k] - U1[0, k-1])/(X1[3, k] - X1[3, k-1]))**2/(N+1)
        cost += W_fxdot   *(1/2)*((U1[1, k] - U1[1, k-1])/(X1[3, k] - X1[3, k-1]))**2/(N+1)
        
        cost += W_deltadot*(1/2)*((U0[0, k] - U0[0, k-1])/(X0[3, k] - X0[3, k-1]))**2/(N+1)
        cost += W_fxdot   *(1/2)*((U0[1, k] - U0[1, k-1])/(X0[3, k] - X0[3, k-1]))**2/(N+1)

    opti.minimize(cost)


    # Define constraints

    # Track bounds
    for k in range(0, N+1):
        opti.subject_to(opti.bounded(-(world.track_width_m_LUT(S[k])/2 - edge_buffer), X0[4, k], (world.track_width_m_LUT(S[k])/2 - edge_buffer)))
        opti.subject_to(opti.bounded(-(world.track_width_m_LUT(S[k])/2 - edge_buffer), X1[4, k], (world.track_width_m_LUT(S[k])/2 - edge_buffer)))


    # Dynamics constraints (trapezoidal integration)
    for k in range(N):
        
        # Compute state derivatives at left knot point
        k0_left, AUX0[0, k], AUX0[1, k], AUX0[2, k], AUX0[3, k], AUX0[4, k], AUX0[5, k], AUX0[6, k], AUX0[7, k], AUX0[8, k], AUX0[9, k], AUX0[10, k], AUX0[11, k] \
            = stm_model.dx_ds_casadi(X0[:, k], U0[:, k], Z[:, k], mu_f_opt[0], mu_r_opt[0], auxiliary_outputs=True)
        k1_left, AUX1[0, k], AUX1[1, k], AUX1[2, k], AUX1[3, k], AUX1[4, k], AUX1[5, k], AUX1[6, k], AUX1[7, k], AUX1[8, k], AUX1[9, k], AUX1[10, k], AUX1[11, k] \
            = stm_model.dx_ds_casadi(X1[:, k], U1[:, k], Z[:, k], mu_f_opt[1], mu_r_opt[1], auxiliary_outputs=True)
        
        # Compute state derivatives at right knot point
        k0_right, AUX0[0, k+1], AUX0[1, k+1], AUX0[2, k+1], AUX0[3, k+1], AUX0[4, k+1], AUX0[5, k+1], AUX0[6, k+1], AUX0[7, k+1], AUX0[8, k+1], AUX0[9, k+1], AUX0[10, k+1], AUX0[11, k+1] \
            = stm_model.dx_ds_casadi(X0[:, k+1], U0[:, k+1], Z[:, k+1], mu_f_opt[0], mu_r_opt[0], auxiliary_outputs=True)
        k1_right, AUX1[0, k+1], AUX1[1, k+1], AUX1[2, k+1], AUX1[3, k+1], AUX1[4, k+1], AUX1[5, k+1], AUX1[6, k+1], AUX1[7, k+1], AUX1[8, k+1], AUX1[9, k+1], AUX1[10, k+1], AUX1[11, k+1] \
            = stm_model.dx_ds_casadi(X1[:, k+1], U1[:, k+1], Z[:, k+1], mu_f_opt[1], mu_r_opt[1], auxiliary_outputs=True)
            
        # Dynamics evolution constraints
        opti.subject_to(X0[:, k+1] - X0[:, k] - 1/2*ds_m*(k0_left + k0_right) == 0)
        opti.subject_to(X1[:, k+1] - X1[:, k] - 1/2*ds_m*(k1_left + k1_right) == 0)

    
    # State feedback control constraints
    opti.subject_to(U1[0, :] == U0[0, :] - fb_gains_dict["K_delta_e"] *(X1[4, :] - X0[4, :])
                                         - fb_gains_dict["K_delta_dpsi"]*(X1[5, :] - X0[5, :]))
    opti.subject_to(U1[1, :] == U0[1, :] - fb_gains_dict["K_fx_ux"]*(X1[0, :] - X0[0, :]))
    
    
    # Actuator constraints
    opti.subject_to(opti.bounded(-(PI/180)*stm_model.params_dict["max_delta_deg"], U0[0, :], (PI/180)*stm_model.params_dict["max_delta_deg"]))
    opti.subject_to(opti.bounded(-(PI/180)*stm_model.params_dict["max_delta_deg"], U1[0, :], (PI/180)*stm_model.params_dict["max_delta_deg"]))
    
    opti.subject_to(U0[1, :]*X0[0, :] <= stm_model.params_dict["p_eng_max_kw"])
    opti.subject_to(U1[1, :]*X1[0, :] <= stm_model.params_dict["p_eng_max_kw"])
    
    opti.subject_to(opti.bounded(-mu_f_opt[0]*AUX0[4, :]*cos(AUX0[9, :]),  AUX0[0, :], mu_f_opt[0]*AUX0[4, :]*cos(AUX0[9, :]))) # Constrain Fx_f command to model 0 friction circle
    opti.subject_to(opti.bounded(-mu_r_opt[0]*AUX0[5, :]*cos(AUX0[10, :]), AUX0[2, :], mu_r_opt[0]*AUX0[5, :]*cos(AUX0[10, :]))) # Constrain Fx_r command to model 0 friction circle

    if(vx_max_mps is not None):
        opti.subject_to(X0[0, :] <= vx_max_mps)
        opti.subject_to(X1[0, :] <= vx_max_mps)


    # Boundary conditions
    if(convergent_lap):
        opti.subject_to(X0[0:3, 0] == X0[0:3, -1])
        opti.subject_to(X0[4:7, 0] == X0[4:7, -1])

        opti.subject_to(X1[0:3, 0] == X1[0:3, -1])
        opti.subject_to(X1[4:7, 0] == X1[4:7, -1])
        
        opti.subject_to(X1[3, 0] == x_0[3])
        opti.subject_to(X0[3, 0] == x_0[3])

        opti.subject_to(U0[0, 0] == U0[0, -1])
        opti.subject_to(U0[1, 0] == U0[1, -1])
        
    else:
        raise NotImplementedError("Trajectory optimization currently is configured for full convergent laps.")


    # Initialize solver
    opti.set_initial(X0, x_warmstart)
    opti.set_initial(X1, x_warmstart)
    opti.set_initial(U0[0, :], u_warmstart[0, :])
    opti.set_initial(U0[1, :], u_warmstart[1, :])
    opti.set_initial(U1[0, :], u_warmstart[0, :])
    opti.set_initial(U1[1, :], u_warmstart[1, :])


    #######################################################################


    # Setup and call optimization
    p_opts = {"expand": True}
    s_opts = {"print_level" : 5, "max_iter" : 5000}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()
    
    # Process results
    X0_sol = sol.value(X0)
    X1_sol = sol.value(X1)

    U0_sol = sol.value(U0)
    U1_sol = sol.value(U1)
    
    Z_sol = Z

    AUX0_sol = sol.value(AUX0)
    AUX1_sol = sol.value(AUX1)

    S_sol = np.linspace(s_0, s_0 + N*ds_m, N+1)
    
    print(f"Solution maneuver time (model 0, model 1): {X0_sol[3, -1]:0.02f}, {X1_sol[3, -1]:0.02f}")
    print(f"Maximum acceleration (model 0, model 1): {1/9.81*np.max(np.sqrt(AUX0_sol[6, :]**2 + AUX0_sol[7, :]**2)):0.02f}, {1/9.81*np.max(np.sqrt(AUX1_sol[6, :]**2 + AUX1_sol[7, :]**2)):0.02f} g's")
    
    X_0_sol, Y_0_sol, Z_0_sol = world.map_match_vectorized(S_sol, X0_sol[4, :])
    X_1_sol, Y_1_sol, Z_1_sol = world.map_match_vectorized(S_sol, X1_sol[4, :])
       
    posE_true_innerbound = world.inner_bounds_posE_m_interp_fcn(S_sol)
    posN_true_innerbound = world.inner_bounds_posN_m_interp_fcn(S_sol)
    posE_true_outerbound = world.outer_bounds_posE_m_interp_fcn(S_sol)
    posN_true_outerbound = world.outer_bounds_posN_m_interp_fcn(S_sol)
    
    posE_virtual_innerbound = world.virtual_inner_bounds_posE_m_interp_fcn(S_sol)
    posN_virtual_innerbound = world.virtual_inner_bounds_posN_m_interp_fcn(S_sol)
    posE_virtual_outerbound = world.virtual_outer_bounds_posE_m_interp_fcn(S_sol)
    posN_virtual_outerbound = world.virtual_outer_bounds_posN_m_interp_fcn(S_sol)
    
    trajectory_0_sol = {
        "X":                         X0_sol,
        "U":                         U0_sol,
        "Z":                         Z_sol,
        "vx_mps":                    X0_sol[0, :],
        "vy_mps":                    X0_sol[1, :],
        "r_radps":                   X0_sol[2, :],
        "t_s":                       X0_sol[3, :],
        "e_m":                       X0_sol[4, :],
        "dpsi_rad":                  X0_sol[5, :],
        "dfz_long_kn":               X0_sol[6, :],
        "x_m":                       X_0_sol,
        "y_m":                       Y_0_sol,
        "z_m":                       Z_0_sol,
        "s_m":                       S_sol,
        "delta_rad":                 U0_sol[0, :],
        "fx_kn":                     U0_sol[1, :],
        "fx_f_kn":                   AUX0_sol[0, :],
        "fy_f_kn":                   AUX0_sol[1, :],
        "fx_r_kn":                   AUX0_sol[2, :],
        "fy_r_kn":                   AUX0_sol[3, :],
        "fz_f_kn":                   AUX0_sol[4, :],
        "fz_r_kn":                   AUX0_sol[5, :],
        "ax_mps2":                   AUX0_sol[6, :],
        "ay_mps2":                   AUX0_sol[7, :],
        "alpha_z_radps2":            AUX0_sol[8, :],
        "alpha_f_rad":               AUX0_sol[9, :],
        "alpha_r_rad":               AUX0_sol[10, :],
        "mz_diff_braking_knm":       AUX0_sol[11, :],
        "posE_m_true_innerbound":    posE_true_innerbound,
        "posN_m_true_innerbound":    posN_true_innerbound,
        "posU_m_true_innerbound":    posN_true_innerbound,
        "posE_m_true_outerbound":    posE_true_outerbound,
        "posN_m_true_outerbound":    posN_true_outerbound,
        "posU_m_true_outerbound":    posN_true_outerbound,
        "posE_m_virtual_innerbound": posE_virtual_innerbound,
        "posN_m_virtual_innerbound": posN_virtual_innerbound,
        "posU_m_virtual_innerbound": posN_virtual_innerbound,
        "posE_m_virtual_outerbound": posE_virtual_outerbound,
        "posN_m_virtual_outerbound": posN_virtual_outerbound,
        "posU_m_virtual_outerbound": posN_virtual_outerbound,
        "stm_model_params":          stm_model.params_dict,
        "final_t_s":                 X0_sol[3, -1],
    }

    trajectory_1_sol = {
        "X":                         X1_sol,
        "U":                         U1_sol,
        "Z":                         Z_sol,
        "vx_mps":                    X1_sol[0, :],
        "vy_mps":                    X1_sol[1, :],
        "r_radps":                   X1_sol[2, :],
        "t_s":                       X1_sol[3, :],
        "e_m":                       X1_sol[4, :],
        "dpsi_rad":                  X1_sol[5, :],
        "dfz_long_kn":               X1_sol[6, :],
        "x_m":                       X_1_sol,
        "y_m":                       Y_1_sol,
        "z_m":                       Z_1_sol,
        "s_m":                       S_sol,
        "delta_rad":                 U1_sol[0, :],
        "fx_kn":                     U1_sol[1, :],
        "fx_f_kn":                   AUX1_sol[0, :],
        "fy_f_kn":                   AUX1_sol[1, :],
        "fx_r_kn":                   AUX1_sol[2, :],
        "fy_r_kn":                   AUX1_sol[3, :],
        "fz_f_kn":                   AUX1_sol[4, :],
        "fz_r_kn":                   AUX1_sol[5, :],
        "ax_mps2":                   AUX1_sol[6, :],
        "ay_mps2":                   AUX1_sol[7, :],
        "alpha_z_radps2":            AUX1_sol[8, :],
        "alpha_f_rad":               AUX1_sol[9, :],
        "alpha_r_rad":               AUX1_sol[10, :],
        "mz_diff_braking_knm":       AUX1_sol[11, :],
        "posE_m_true_innerbound":    posE_true_innerbound,
        "posN_m_true_innerbound":    posN_true_innerbound,
        "posU_m_true_innerbound":    posN_true_innerbound,
        "posE_m_true_outerbound":    posE_true_outerbound,
        "posN_m_true_outerbound":    posN_true_outerbound,
        "posU_m_true_outerbound":    posN_true_outerbound,
        "posE_m_virtual_innerbound": posE_virtual_innerbound,
        "posN_m_virtual_innerbound": posN_virtual_innerbound,
        "posU_m_virtual_innerbound": posN_virtual_innerbound,
        "posE_m_virtual_outerbound": posE_virtual_outerbound,
        "posN_m_virtual_outerbound": posN_virtual_outerbound,
        "posU_m_virtual_outerbound": posN_virtual_outerbound,
        "stm_model_params":          stm_model.params_dict,
        "final_t_s":                 X1_sol[3, -1],
    }

    return trajectory_0_sol, trajectory_1_sol
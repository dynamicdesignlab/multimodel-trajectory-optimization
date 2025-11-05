import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_results(results):
    
    plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    ref_color = plot_colors[2]
    ref_ls   = "dashed"
    ref_ls_2 = "dotted"
    
    
    sim_cmap = plt.colormaps["plasma"]
    cmap_as_array = sim_cmap(np.arange(1024)/(1024-1))
    cmap_as_array = cmap_as_array[int(0*len(cmap_as_array)):int(0.80*len(cmap_as_array)), :]
    sim_cmap = LinearSegmentedColormap.from_list("plasma_slice", cmap_as_array)
    
    sim_color = "black"
    sim_ls = "-"
    sim_lw = 1
    sim_alpha = 0.5
    

    # Allocate figures
    fig0, ax0 = plt.subplots(num="Trajectory overhead", figsize = (5.5, 5.5), constrained_layout=True)  
    fig1, ax1 = plt.subplots(4, 2, num="Vehicle states", figsize = (5.5, 5.5), constrained_layout=True)
    fig2, ax2 = plt.subplots(2, 1, num="Vehicle control inputs", figsize = (5.5, 5.5), constrained_layout=True)
    
    # Model 0 solution
    # Plot the trajectory overhead
    ax0.plot(results["trajectory_0_sol"]["x_m"], results["trajectory_0_sol"]["y_m"], zorder=3, color=ref_color, ls=ref_ls)
    ax0.plot(results["trajectory_0_sol"]["posE_m_true_innerbound"], results["trajectory_0_sol"]["posN_m_true_innerbound"], color = "gray")
    ax0.plot(results["trajectory_0_sol"]["posE_m_true_outerbound"], results["trajectory_0_sol"]["posN_m_true_outerbound"], color = "gray")
    ax0.plot(results["trajectory_0_sol"]["posE_m_virtual_innerbound"], results["trajectory_0_sol"]["posN_m_virtual_innerbound"], color = "gray", alpha = .5, linestyle = '--')
    ax0.plot(results["trajectory_0_sol"]["posE_m_virtual_outerbound"], results["trajectory_0_sol"]["posN_m_virtual_outerbound"], color = "gray", alpha = .5, linestyle = '--')
    ax0.set_xlabel("East [m]")
    ax0.set_ylabel("North [m]") 
    ax0.set_aspect("equal")    
    
    # Plot the states
    ax1[0, 0].plot(results["trajectory_0_sol"]["s_m"], results["trajectory_0_sol"]["vx_mps"], zorder=3, color = ref_color, ls=ref_ls)
    ax1[1, 0].plot(results["trajectory_0_sol"]["s_m"], results["trajectory_0_sol"]["vy_mps"], zorder=3, color = ref_color, ls=ref_ls)
    ax1[2, 0].plot(results["trajectory_0_sol"]["s_m"], results["trajectory_0_sol"]["r_radps"], zorder=3, color = ref_color, ls=ref_ls)
    ax1[0, 1].plot(results["trajectory_0_sol"]["s_m"], results["trajectory_0_sol"]["t_s"], zorder=3, color = ref_color, ls=ref_ls)
    ax1[1, 1].plot(results["trajectory_0_sol"]["s_m"], results["trajectory_0_sol"]["e_m"], zorder=3, color = ref_color, ls=ref_ls)
    ax1[2, 1].plot(results["trajectory_0_sol"]["s_m"], results["trajectory_0_sol"]["dpsi_rad"], zorder=3, color = ref_color, ls=ref_ls)
    ax1[3, 0].plot(results["trajectory_0_sol"]["s_m"], results["trajectory_0_sol"]["dfz_long_kn"], zorder=3, color = ref_color, ls=ref_ls)

    ax1[0, 0].set_ylabel(r"$v_x$ [m/s]")
    ax1[1, 0].set_ylabel(r"$v_y$ [m/s]")
    ax1[2, 0].set_ylabel(r"$r$ [rad/s]")
    ax1[0, 1].set_ylabel(r"$t$ [s]")
    ax1[1, 1].set_ylabel(r"$e$ [m]")
    ax1[2, 1].set_ylabel(r"$\Delta\psi$ [rad]")
    ax1[3, 0].set_ylabel(r"$\Delta F_{z,\mathrm{long}}$ [rad]")
    ax1[2, 0].set_xlabel(r"$s$ [m]")
    ax1[2, 1].set_xlabel(r"$s$ [m]")
    fig1.align_ylabels()
    
    # Plot the control inputs
    ax2[0].plot(results["trajectory_0_sol"]["s_m"], results["trajectory_0_sol"]["delta_rad"], zorder=3, color = ref_color, ls=ref_ls)
    ax2[1].plot(results["trajectory_0_sol"]["s_m"], results["trajectory_0_sol"]["fx_kn"], zorder=3, color = ref_color, ls=ref_ls)
    ax2[0].set_ylabel(r"$\delta^\mathrm{cmd}$ [rad]")
    ax2[1].set_ylabel(r"$F_x^\mathrm{cmd}$ [kn]")
    ax2[1].set_xlabel("s [m]")
    fig2.align_ylabels()
    

    
    # Model 1 solution, if applicable
    if("trajectory_1_sol" in results):        
        ax0.plot(results["trajectory_1_sol"]["x_m"], results["trajectory_1_sol"]["y_m"], zorder=3, color=ref_color, ls=ref_ls_2)
        
        ax1[0, 0].plot(results["trajectory_1_sol"]["s_m"], results["trajectory_1_sol"]["vx_mps"], zorder=3, color=ref_color, ls=ref_ls_2)
        ax1[1, 0].plot(results["trajectory_1_sol"]["s_m"], results["trajectory_1_sol"]["vy_mps"], zorder=3, color=ref_color, ls=ref_ls_2)
        ax1[2, 0].plot(results["trajectory_1_sol"]["s_m"], results["trajectory_1_sol"]["r_radps"], zorder=3, color=ref_color, ls=ref_ls_2)
        ax1[0, 1].plot(results["trajectory_1_sol"]["s_m"], results["trajectory_1_sol"]["t_s"], zorder=3, color=ref_color, ls=ref_ls_2)
        ax1[1, 1].plot(results["trajectory_1_sol"]["s_m"], results["trajectory_1_sol"]["e_m"], zorder=3, color=ref_color, ls=ref_ls_2)
        ax1[2, 1].plot(results["trajectory_1_sol"]["s_m"], results["trajectory_1_sol"]["dpsi_rad"], zorder=3, color=ref_color, ls=ref_ls_2)
        ax1[3, 0].plot(results["trajectory_1_sol"]["s_m"], results["trajectory_1_sol"]["dpsi_rad"], zorder=3, color=ref_color, ls=ref_ls_2)
        
        ax2[0].plot(results["trajectory_1_sol"]["s_m"], results["trajectory_1_sol"]["delta_rad"], zorder=3, color=ref_color, ls=ref_ls_2)
        ax2[1].plot(results["trajectory_1_sol"]["s_m"], results["trajectory_1_sol"]["fx_kn"], zorder=3, color=ref_color, ls=ref_ls_2)
        
    
    # Simulation results, if applicable
    keys = results.keys()
    n_sims = 0
    for key in keys:
        if("simulation" in key):
            n_sims += 1
    
    sim_counter = 0
    for key in keys:
        if("simulation" in key):
            color = sim_cmap(sim_counter/(n_sims))
            color = sim_color
            
            ax0.plot(results[key]["east_m"], results[key]["north_m"], zorder=1, color=color, ls=sim_ls, lw=sim_lw, alpha=sim_alpha)
            
            ax1[0, 0].plot(results[key]["s_sim"], results[key]["x_sim"][0, :], zorder=1, color=color, ls=sim_ls, lw=sim_lw, alpha=sim_alpha)
            ax1[1, 0].plot(results[key]["s_sim"], results[key]["x_sim"][1, :], zorder=1, color=color, ls=sim_ls, lw=sim_lw, alpha=sim_alpha)
            ax1[2, 0].plot(results[key]["s_sim"], results[key]["x_sim"][2, :], zorder=1, color=color, ls=sim_ls, lw=sim_lw, alpha=sim_alpha)
            ax1[0, 1].plot(results[key]["s_sim"], results[key]["x_sim"][3, :], zorder=1, color=color, ls=sim_ls, lw=sim_lw, alpha=sim_alpha)
            ax1[1, 1].plot(results[key]["s_sim"], results[key]["x_sim"][4, :], zorder=1, color=color, ls=sim_ls, lw=sim_lw, alpha=sim_alpha)
            ax1[2, 1].plot(results[key]["s_sim"], results[key]["x_sim"][5, :], zorder=1, color=color, ls=sim_ls, lw=sim_lw, alpha=sim_alpha)
            ax1[3, 0].plot(results[key]["s_sim"], results[key]["x_sim"][6, :], zorder=1, color=color, ls=sim_ls, lw=sim_lw, alpha=sim_alpha)
            
            ax2[0].plot(results[key]["s_sim"], results[key]["u_sim"][0, :], zorder=1, color=color, ls=sim_ls, lw=sim_lw, alpha=sim_alpha)
            ax2[1].plot(results[key]["s_sim"], results[key]["u_sim"][1, :], zorder=1, color=color, ls=sim_ls, lw=sim_lw, alpha=sim_alpha)
            
            sim_counter += 1
    
    return fig0, fig1, fig2
